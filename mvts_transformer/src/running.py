import logging
import sys
import os
import traceback
import json
from datetime import datetime
import string
import random
from collections import OrderedDict
import time
import pickle
from functools import partial

import ipdb
import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn

from utils import utils, analysis, visualization_utils
from models.loss import l1_reg_loss, l2_reg_loss
from datasets.dataset import (
    ImputationDataset,
    TransductionDataset,
    ClassiregressionDataset,
    collate_unsuperv,
    collate_superv,
)
import matplotlib.pyplot as plt

logger = logging.getLogger("__main__")

NEG_METRICS = {"loss"}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    task = config["task"]
    baseline = config["baseline"]

    if task == "imputation":
        if (config["mask_distribution"] == "early" or config["mask_distribution"] == "autoregressive"):
            return partial(ImputationDataset, mean_mask_length=config['mean_mask_length'],
                           masking_ratio=config['proportion'], mode=config['mask_mode'],
                           distribution='early', exclude_feats=config['exclude_feats']),\
                collate_unsuperv, UnsupervisedRunner
        else:
            return partial(ImputationDataset, mean_mask_length=config['mean_mask_length'],
                           masking_ratio=config['masking_ratio'], mode=config['mask_mode'],
                           distribution=config['mask_distribution'], exclude_feats=config['exclude_feats']),\
                collate_unsuperv, UnsupervisedRunner
    if task == "transduction":
        return partial(TransductionDataset, mask_feats=config['mask_feats'],
                       start_hint=config['start_hint'], end_hint=config['end_hint']), collate_unsuperv, UnsupervisedRunner
    if (task == "classification") or (task == "regression"):
        return ClassiregressionDataset, collate_superv, SupervisedRunner
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical(
                "Failed to load configuration file. Check JSON syntax and verify that files exist"
            )
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config["output_dir"]
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(
                output_dir
            )
        )

    output_dir = os.path.join(output_dir, config["experiment_name"])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config["initial_timestamp"] = formatted_timestamp
    if (not config["no_timestamp"]) or (len(config["experiment_name"]) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config["output_dir"] = output_dir
    config["plot_dir"] = os.path.join(output_dir, "plots")
    config["save_dir"] = os.path.join(output_dir, "checkpoints")
    config["pred_dir"] = os.path.join(output_dir, "predictions")
    config["tensorboard_dir"] = os.path.join(output_dir, "tb_summaries")
    utils.create_dirs(
        [config["plot_dir"], config["save_dir"], config["pred_dir"], config["tensorboard_dir"]]
    )

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, "configuration.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def fold_evaluate(
    dataset, model, device, loss_module, target_feats, config, dataset_name
):
    allfolds = {
        "target_feats": target_feats,  # list of len(num_folds), each element: list of target feature integer indices
        # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) prediction per sample
        "predictions": [],
        # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) target/original input per sample
        "targets": [],
        # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) boolean mask per sample
        "target_masks": [],
        # list of len(num_folds), each element: (num_samples, num_metrics) metric per sample
        "metrics": [],
        "IDs": [],
    }  # list of len(num_folds), each element: (num_samples,) ID per sample

    for i, tgt_feats in enumerate(target_feats):
        dataset.mask_feats = tgt_feats  # set the transduction target features

        loader = DataLoader(
            dataset=dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_unsuperv(x, max_len=config["max_seq_len"]),
        )

        evaluator = UnsupervisedRunner(
            model,
            loader,
            device,
            loss_module,
            print_interval=config["print_interval"],
            console=config["console"],
        )

        logger.info(
            "Evaluating {} set, fold: {}, target features: {}".format(
                dataset_name, i, tgt_feats
            )
        )
        aggr_metrics, per_batch = evaluate(evaluator)

        metrics_array = convert_metrics_per_batch_to_per_sample(
            per_batch["metrics"], per_batch["target_masks"]
        )
        metrics_array = np.concatenate(metrics_array, axis=0)
        allfolds["metrics"].append(metrics_array)
        allfolds["predictions"].append(np.concatenate(per_batch["predictions"], axis=0))
        allfolds["targets"].append(np.concatenate(per_batch["targets"], axis=0))
        allfolds["target_masks"].append(
            np.concatenate(per_batch["target_masks"], axis=0)
        )
        allfolds["IDs"].append(np.concatenate(per_batch["IDs"], axis=0))

        metrics_mean = np.mean(metrics_array, axis=0)
        metrics_std = np.std(metrics_array, axis=0)
        for m, metric_name in enumerate(list(aggr_metrics.items())[1:]):
            logger.info(
                "{}:: Mean: {:.3f}, std: {:.3f}".format(
                    metric_name, metrics_mean[m], metrics_std[m]
                )
            )

    pred_filepath = os.path.join(
        config["pred_dir"], dataset_name + "_fold_transduction_predictions.pickle"
    )
    logger.info("Serializing predictions into {} ... ".format(pred_filepath))
    with open(pred_filepath, "wb") as f:
        pickle.dump(allfolds, f, pickle.HIGHEST_PROTOCOL)


def convert_metrics_per_batch_to_per_sample(metrics, target_masks):
    """
    Args:
        metrics: list of len(num_batches), each element: list of len(num_metrics), each element: (num_active_in_batch,) metric per element
        target_masks: list of len(num_batches), each element: (batch_size, seq_len, feat_dim) boolean mask: 1s active, 0s ignore
    Returns:
        metrics_array = list of len(num_batches), each element: (batch_size, num_metrics) metric per sample
    """
    metrics_array = []
    for b, batch_target_masks in enumerate(target_masks):
        num_active_per_sample = np.sum(batch_target_masks, axis=(1, 2))
        # (num_active_in_batch, num_metrics)
        batch_metrics = np.stack(metrics[b], axis=1)
        ind = 0
        metrics_per_sample = np.zeros(
            (len(num_active_per_sample), batch_metrics.shape[1])
        )  # (batch_size, num_metrics)
        for n, num_active in enumerate(num_active_per_sample):
            new_ind = ind + num_active
            metrics_per_sample[n, :] = np.sum(batch_metrics[ind:new_ind, :], axis=0)
            ind = new_ind
        metrics_array.append(metrics_per_sample)
    return metrics_array


def evaluate(evaluator):
    """Perform a single, one-off evaluation on an evaluator object (initialized with a dataset)"""

    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = evaluator.evaluate(epoch_num=None, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    print()
    print_str = "Evaluation Summary: "
    for k, v in aggr_metrics.items():
        if v is not None:
            print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)
    logger.info(
        "Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(eval_runtime)
        )
    )

    return aggr_metrics, per_batch


def validate(
    val_evaluator,
    tensorboard_writer,
    config,
    best_metrics,
    best_value,
    epoch,
    keep_predictions=False,
    require_padding=False,
    need_attn_weights=False,
):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        if keep_predictions:
            aggr_metrics, per_batch, predictions, targets = val_evaluator.evaluate(
                epoch,
                config,
                keep_predictions=True,
                require_padding=require_padding,
                keep_all=True,
                need_attn_weights=need_attn_weights
            )
        else:
            aggr_metrics, per_batch = val_evaluator.evaluate(epoch, config, keep_all=True, require_padding=require_padding, need_attn_weights=need_attn_weights)

    eval_runtime = time.time() - eval_start_time
    logger.info(
        "Validation runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(eval_runtime)
        )
    )

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    logger.info(
        "Avg val. time: {} hours, {} minutes, {} seconds".format(
            *utils.readable_time(avg_val_time)
        )
    )
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = "Epoch {} Validation Summary: ".format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar("{}/val".format(k), v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)

    if config["key_metric"] in NEG_METRICS:
        condition = aggr_metrics[config["key_metric"]] < best_value
    else:
        condition = aggr_metrics[config["key_metric"]] > best_value
    if condition:
        best_value = aggr_metrics[config["key_metric"]]
        utils.save_model(
            os.path.join(config["save_dir"], "model_best.pth"),
            epoch,
            val_evaluator.model,
        )

        # TODO: integrate from pre-training to fine-tuning
        # with open(os.path.join("./experiments", config['experiment_name'] + "_model_path.txt"), "w") as f:
        #     f.write(os.path.join(config['save_dir'], 'model_best.pth'))

        # @joshuafan: savez doesn't work, try concatenating "per_batch" predictions/targets first into single numpy array (instead of list of batches)
        per_batch["targets"] = np.concatenate(per_batch["targets"], axis=0)
        per_batch["predictions"] = np.concatenate(per_batch["predictions"], axis=0)
        per_batch["metrics"] = np.concatenate(per_batch["metrics"], axis=0)
        per_batch["IDs"] = np.concatenate(per_batch["IDs"], axis=0)

        best_metrics = aggr_metrics.copy()

        pred_filepath = os.path.join(config["pred_dir"], "best_predictions")
        np.savez(pred_filepath, **per_batch)

    if keep_predictions:
        return aggr_metrics, best_metrics, best_value, predictions, targets

    return aggr_metrics, best_metrics, best_value


def check_progress(epoch):
    if epoch in [100, 140, 160, 220, 280, 340]:
        return True
    else:
        return False


class BaseRunner(object):
    def __init__(
        self,
        model,
        dataloader,
        device,
        loss_module,
        optimizer=None,
        l1_reg=None,
        l2_reg=None,
        print_interval=10,
        console=True,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None, keep_predictions=False, require_padding=False, use_smoothing=False, smoothing_lambda=0, need_attn_weights=False):
        raise NotImplementedError("Please override in child class")

    def evaluate(
        self,
        epoch_num=None,
        config=None,
        keep_predictions=False,
        require_padding=False,
        keep_all=True,
        need_attn_weights=False
    ):
        raise NotImplementedError("Please override in child class")

    def print_callback(self, i_batch, metrics, prefix=""):
        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnsupervisedRunner(BaseRunner):
    def train_epoch(self, epoch_num=None, keep_predictions=False, require_padding=False, use_smoothing=False, smoothing_lambda=0, need_attn_weights=False):
        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        all_predictions, all_targets = [], []

        for i, batch in enumerate(self.dataloader):
            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            # 1s: mask and predict, 0s: unaffected input (ignore)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # (batch_size, padded_length, feat_dim)
            if require_padding:
                if need_attn_weights:
                    predictions, attn_weights_layers = self.model(X.to(self.device), padding_masks)
                else:
                    predictions = self.model(X.to(self.device), padding_masks)
            else:
                if need_attn_weights:
                    predictions, attn_weights_layers = self.model(X.to(self.device))
                else:
                    predictions = self.model(X.to(self.device))

            all_predictions.append(predictions)
            all_targets.append(targets)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            # (num_active,) individual loss (square error per element) for each active value in batch
            loss = self.loss_module(predictions, targets, target_masks)
            batch_loss = torch.sum(loss)
            # mean loss (over active elements) used for optimization
            mean_loss = batch_loss / len(loss)

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            if use_smoothing:
                attn_smoothness_loss = 0
                for attn_weights in attn_weights_layers:
                  attn_smoothness_loss += torch.sum((attn_weights[:, 1:] - attn_weights[:, :-1]) ** 2)
                total_loss += smoothing_lambda * attn_smoothness_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        # average loss per element for whole epoch
        epoch_loss = epoch_loss / total_active_elements
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        if keep_predictions:
            return self.epoch_metrics, all_predictions, all_targets
        return self.epoch_metrics

    def evaluate(
        self,
        epoch_num=None,
        config=None,
        keep_predictions=False,
        require_padding=False,
        keep_all=True,
        need_attn_weights=False
    ):
        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        if keep_all:
            per_batch = {
                "target_masks": [],
                "targets": [],
                "predictions": [],
                "metrics": [],
                "IDs": [],
            }
        for i, batch in enumerate(self.dataloader):
            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            # 1s: mask and predict, 0s: unaffected input (ignore)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # (batch_size, padded_length, feat_dim)
            if require_padding:
                if need_attn_weights:
                    predictions, _ = self.model(X.to(self.device), padding_masks)
                else:
                    predictions = self.model(X.to(self.device), padding_masks)
            else:
                if need_attn_weights:
                    predictions, _ = self.model(X.to(self.device))
                else:
                    predictions = self.model(X.to(self.device))

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            # (num_active,) individual loss (square error per element) for each active value in batch
            loss = self.loss_module(predictions, targets, target_masks)
            batch_loss = torch.sum(loss).cpu().item()
            # mean loss (over active elements) used for optimization the batch
            mean_loss = batch_loss / len(loss)

            if keep_all:
                per_batch["target_masks"].append(target_masks.cpu().detach().numpy())
                per_batch["targets"].append(targets.cpu().detach().numpy())
                per_batch["predictions"].append(predictions.cpu().detach().numpy())
                per_batch["metrics"].append(loss.cpu().detach().numpy())
                per_batch["IDs"].append(IDs)

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            total_active_elements += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        # average loss per element for whole epoch
        epoch_loss = epoch_loss / total_active_elements
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class SupervisedRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(SupervisedRunner, self).__init__(*args, **kwargs)

        if isinstance(args[3], torch.nn.CrossEntropyLoss):
            self.classification = True  # True if classification, False if regression
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        else:
            self.classification = False

    def train_epoch(self, config, epoch_num=None, keep_predictions=False, require_padding=False, use_smoothing=False, smoothing_lambda=0, need_attn_weights=False):
        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        supervised_loss, supervised_smoothing_loss, posenc_loss = 0, 0, 0
        all_predictions, all_targets = [], []

        for i, batch in enumerate(self.dataloader):
            X, targets, padding_masks, IDs = batch  # @joshuafan added time
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits

            # Plot dir if needed
            if i == 0 and epoch_num % 100 == 0:
                plot_dir = os.path.join(config['plot_dir'], f'train_epoch{epoch_num}')
                os.makedirs(plot_dir, exist_ok=True)
            else:
                plot_dir = None

            if config["mixtype"] != 'none':
                X, targets = utils.generate_mixup_data(config, X, targets, self.device)

            if require_padding:
                if need_attn_weights:
                    predictions, attn_weights_layers = self.model(X.to(self.device), padding_masks, plot_dir=plot_dir)
                else:
                    predictions = self.model(X.to(self.device), padding_masks, plot_dir=plot_dir)
            else:
                if need_attn_weights:
                    predictions, attn_weights_layers = self.model(X.to(self.device), plot_dir=plot_dir)
                else:
                    predictions = self.model(X.to(self.device), plot_dir=plot_dir)

            if config['normalize_label']:
                predictions = predictions * config["label_std"] + config["label_mean"]
            all_predictions.append(predictions.detach().flatten())
            all_targets.append(targets.detach().flatten())

            # (batch_size,) loss for each sample in the batch
            loss = self.loss_module(predictions, targets)
            batch_loss = torch.sum(loss)
            # mean loss (over samples) used for optimization
            mean_loss = batch_loss / len(loss)

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            if self.l1_reg:
                total_loss += (self.l1_reg * l1_reg_loss(self.model))

            supervised_loss += batch_loss.item()  # total_loss.cpu().detach().numpy()

            if use_smoothing:  # attn_weights_layers: [batch, n_layers*n_heads, seq_len, seq_len]

                attn_smoothness_loss = 0
                attn_weights_layers = attn_weights_layers.reshape(-1, attn_weights_layers.shape[2], attn_weights_layers.shape[3])   # Convert to [something, seq_len, seq_len] - list of attention matrices
                # attn_smoothness_loss = ((attn_weights_layers[:, :, 1:] - attn_weights_layers[:, :, :-1]) ** 2).sum(dim=2).mean()
                attn_smoothness_loss = ((attn_weights_layers[:, :, 2:] + attn_weights_layers[:, :, :-2] - 2*attn_weights_layers[:, :, 1:-1]) ** 2).sum(dim=2).mean()
                total_loss += smoothing_lambda * attn_smoothness_loss
            else:
                attn_smoothness_loss = torch.tensor(0)

            supervised_smoothing_loss += (attn_smoothness_loss.item() * smoothing_lambda * len(loss))  # put in same scale as batch_loss

            # Positional encoding smoothness loss. TODO - we should also save it so we can plot
            if (config["model"] == "climax_smooth") and (('learnable' in config['pos_encoding']) or (config['relative_pos_encoding'] == 'erpe')):
                # if config['lambda_posenc_smoothness'] > 0:
                posenc_loss_batch = self.model.posenc_smoothness_loss(logger, plot_dir=plot_dir, epoch_num=epoch_num)
                total_loss += config['lambda_posenc_smoothness'] * posenc_loss_batch
                posenc_loss += config['lambda_posenc_smoothness'] * posenc_loss_batch.item() * len(loss)  # put in same scale as batch_loss
            else:
                assert config['lambda_posenc_smoothness'] == 0

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        # average loss per sample for whole epoch
        epoch_loss = epoch_loss / total_samples
        supervised_loss = supervised_loss / total_samples
        posenc_loss = posenc_loss / total_samples

        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        if keep_predictions:
            return self.epoch_metrics, torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0), supervised_loss, supervised_smoothing_loss, posenc_loss

        return self.epoch_metrics

    def evaluate(self, epoch_num=None, config=None, keep_predictions=False, require_padding=False, keep_all=True, plot=False, need_attn_weights=False):
        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        all_predictions, all_targets = [], []

        per_batch = {
            "target_masks": [],
            "targets": [],
            "predictions": [],
            "metrics": [],
            "IDs": [],
        }

        for i, batch in enumerate(self.dataloader):
            X, targets, padding_masks, IDs = batch
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits

            # Plot dir if needed
            if i == 0 and epoch_num % 100 == 0 and config is not None:
                plot_dir = os.path.join(config['plot_dir'], f'val_epoch{epoch_num}')
                os.makedirs(plot_dir, exist_ok=True)
            else:
                plot_dir = None

            if require_padding:
                if need_attn_weights:
                    predictions, _ = self.model(X.to(self.device), padding_masks, plot_dir=plot_dir)
                else:
                    predictions = self.model(X.to(self.device), padding_masks, plot_dir=plot_dir)
            else:
                if need_attn_weights:
                    predictions, _ = self.model(X.to(self.device), plot_dir=plot_dir)
                else:
                    predictions = self.model(X.to(self.device), plot_dir=plot_dir)

            if config['normalize_label']:
                predictions = predictions * config["label_std"] + config["label_mean"]
            all_predictions.append(predictions.flatten().cpu().detach().numpy())
            all_targets.append(targets.flatten().cpu().detach().numpy())

            # (batch_size,) loss for each sample in the batch
            loss = self.loss_module(predictions, targets)
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch["targets"].append(targets.cpu().numpy())
            per_batch["predictions"].append(predictions.cpu().numpy())
            per_batch["metrics"].append(loss.cpu().numpy())
            per_batch["IDs"].append(np.array(IDs))

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        # average loss per element for whole epoch
        epoch_loss = epoch_loss / total_samples
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        if self.classification:
            predictions = torch.from_numpy(
                np.concatenate(per_batch["predictions"], axis=0)
            )
            # (total_samples, num_classes) est. prob. for each class and sample
            probs = torch.nn.functional.softmax(predictions)
            # (total_samples,) int class index for each sample
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            probs = probs.cpu().numpy()
            targets = np.concatenate(per_batch["targets"], axis=0).flatten()
            # TODO: temporary until I decide how to pass class names
            class_names = np.arange(probs.shape[1])
            metrics_dict = self.analyzer.analyze_classification(
                predictions, targets, class_names
            )

            # same as average recall over all classes
            self.epoch_metrics["accuracy"] = metrics_dict["total_accuracy"]
            # average precision over all classes
            self.epoch_metrics["precision"] = metrics_dict["prec_avg"]

            if self.model.num_classes == 2:
                false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(
                    targets, probs[:, 1]
                )  # 1D scores needed
                self.epoch_metrics["AUROC"] = sklearn.metrics.auc(
                    false_pos_rate, true_pos_rate
                )

                prec, rec, _ = sklearn.metrics.precision_recall_curve(
                    targets, probs[:, 1]
                )
                self.epoch_metrics["AUPRC"] = sklearn.metrics.auc(rec, prec)

        if plot:
            plt.plot(range(len(all_predictions)), all_predictions, label="Predictions", marker="o")
            plt.plot(range(len(all_targets)), all_targets, label="Targets", marker="o")
            plt.legend(["Predictions", "Targets"])
            plt.ylabel("Values")
            plt.xlabel("Time step")
            plt.title("Transformer accuracy")

            if epoch_num is not None and epoch_num % 10 == 0:
                plt.savefig("graphs/AppliancesEnergy/eval_accuracy_epoch_" + str(epoch_num) + ".png")
            else:
                plt.savefig("graphs/AppliancesEnergy/eval_accuracy.png")
            plt.close()

        if keep_predictions:
            return self.epoch_metrics, per_batch, np.concatenate(all_predictions, axis=0), np.concatenate(all_targets, axis=0)

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics
