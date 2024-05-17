"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import random
from optimizers import get_optimizer
from models.loss import get_loss_module
from datasets.datasplit import split_dataset
from datasets.data import data_factory, Normalizer
from datasets.utils import process_data
from utils import utils
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from options import Options
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import json
import pickle
import time
import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")

# 3rd party packages

# Project modules


def main(config):
    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)

    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))  # command used to run

    if config["seed"] is not None:
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config["gpu"] != "-1") else "cpu"
    )
    config['device'] = device
    logger.info("Using device: {}".format(device))
    if device == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    if config["model"] is not None and config["model"] == "swin":
        from models.ts_swin import model_factory
    elif config["model"] is not None and config["model"] == "swin_pool":
        from models.ts_swin_pool import model_factory
    elif config["model"] is not None and config["model"] == "smooth":
        from models.ts_smooth_transformer import model_factory
    elif config["model"] is not None and config["model"] == "patch":
        from models.patch_tst import model_factory
    elif config["model"] is not None and config["model"] == "climax":
        from models.ts_climax_base import model_factory
    elif config["model"] is not None and config["model"] == "climax_smooth":
        from models.ts_climax import model_factory
    elif config["model"] is not None and config["model"] == "convit":
        from models.climax_convit import model_factory
    elif config["model"] is not None and config["model"] == "convit_smooth":
        from models.climax_convit_smooth import model_factory
    elif config["model"] is not None and config["model"] == "convit_2":
        from models.climax_with_convit_blocks import model_factory
    else:
        from models.ts_transformer import model_factory

    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config["data_class"]]
    my_data = data_class(
        config["data_dir"],
        pattern=config["pattern"],
        n_proc=config["n_proc"],
        limit_size=config["limit_size"],
        config=config,
    )

    # Trim data for early prediction
    proportion = config["proportion"]
    length = 144
    if config["baseline"] == 1 and config["proportion"]:
        trimmed_train_data = my_data.feature_df.iloc[: int(proportion * length)]
        # every group of 365
        for i in range(1, int(my_data.feature_df.shape[0] / length)):
            trimmed_train_data = pd.concat(
                [
                    trimmed_train_data,
                    my_data.feature_df.iloc[
                        i * length : int((i + proportion) * length)
                    ],
                ]
            )

        my_data.all_df = trimmed_train_data
        my_data.feature_df = trimmed_train_data

    if config["task"] == "classification":
        validation_method = "StratifiedShuffleSplit"
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = "ShuffleSplit"
        labels = None

    # Split dataset
    if config['test_pattern'] is None:
        config['test_pattern'] = 'TEST'
    test_indices = None

    val_data = my_data
    val_indices = []
    if config["test_pattern"]:  # used if test data come from different files / file patterns
        test_data = data_class(config["data_dir"], pattern=config["test_pattern"], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
    if config["test_from"]:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`. Can work together with `test_pattern`
        test_indices = list(set([line.rstrip() for line in open(config["test_from"]).readlines()]))
        try:
            test_indices = [int(ind) for ind in test_indices]  # integer indices
        except ValueError:
            pass  # in case indices are non-integers
        logger.info(
            "Loaded {} test IDs from file: '{}'".format(
                len(test_indices), config["test_from"]
            )
        )

    if config["val_pattern"]:  # used if val data come from different files / file patterns
        val_data = data_class(config["data_dir"], pattern=config["val_pattern"], n_proc=-1, config=config)
        if config["baseline"] == 2 and config["proportion"]:
            trimmed_val_data = None

            # every group of 365
            for i in range(int(val_data.feature_df.shape[0] / length)):
                mean_per_loc = val_data.feature_df.iloc[
                    i * length : (i + 1) * length
                ].mean(axis=0)
                nrows = int((i + proportion) * length) - i * length
                mean_replicated = pd.DataFrame(
                    np.repeat(
                        np.reshape(mean_per_loc.values, (1, 24)), length - nrows, axis=0
                    )
                )
                mean_replicated.columns = val_data.feature_df.columns
                trimmed_val_data = pd.concat(
                    [
                        trimmed_val_data,
                        val_data.feature_df.iloc[
                            i * length : int((i + proportion) * length)
                        ],
                        mean_replicated,
                    ]
                )

            val_data.all_df = trimmed_val_data
            val_data.feature_df = trimmed_val_data

        val_indices = val_data.all_IDs

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
    if config["val_ratio"] > 0:
        train_indices, val_indices = split_dataset(
            data_indices=my_data.all_IDs,
            validation_method=validation_method,
            n_splits=1,
            validation_ratio=config["val_ratio"],
            random_seed=1337,
            labels=labels,
        )
        # `split_dataset` returns a list of indices *per fold/split*
        train_indices = train_indices[0]
        # `split_dataset` returns a list of indices *per fold/split*
        val_indices = val_indices[0]
    else:
        train_indices = my_data.all_IDs
        if test_indices is None:
            test_indices = []

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    if config["val_pattern"] != "TEST":
      assert len(set(train_indices) & set(val_indices)) == 0

    with open(os.path.join(config["output_dir"], "data_indices.json"), "w") as f:
        try:
            json.dump(
                {
                    "train_indices": list(map(int, train_indices)),
                    "val_indices": list(map(int, val_indices)),
                    "test_indices": list(map(int, test_indices)),
                },
                f,
                indent=4,
            )
        except ValueError:  # in case indices are non-integers
            json.dump(
                {
                    "train_indices": list(train_indices),
                    "val_indices": list(val_indices),
                    "test_indices": list(test_indices),
                },
                f,
                indent=4,
            )

    # Pre-process features
    normalizer = None
    if config["norm_from"]:
        with open(config["norm_from"], "rb") as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    elif config["normalization"] is not None:
        normalizer = Normalizer(config["normalization"])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(
            my_data.feature_df.loc[train_indices]
        )
        if not config["normalization"].startswith("per_sample"):
            # get normalizing values from training set and store for future use
            norm_dict = normalizer.__dict__
            with open(
                os.path.join(config["output_dir"], "normalization.pickle"), "wb"
            ) as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(
                val_data.feature_df.loc[val_indices]
            )
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(
                test_data.feature_df.loc[test_indices]
            )

    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, my_data)

    if config["freeze"]:
        for name, param in model.named_parameters():
            if name.startswith("output_layer"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info(
        "Trainable parameters: {}".format(utils.count_parameters(model, trainable=True))
    )

    # Initialize optimizer

    if config["global_reg"]:
        weight_decay = config["l2_reg"]
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config["l2_reg"]

    optim_class = get_optimizer(config["optimizer"])
    optimizer = optim_class(
        model.parameters(), lr=config["lr"], weight_decay=weight_decay
    )

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config["lr"]  # current learning step
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(
            model,
            config["load_model"],
            optimizer,
            config["resume"],
            config["change_output"],
            config["lr"],
            config["lr_step"],
            config["lr_factor"],
        )
    model.to(device)

    loss_module = get_loss_module(config)

    require_padding = (
        config["model"] is None
        or config["model"] == "transformer"
        or (
            config["model"] == "swin"
            and (config["task"] is None or config["task"] == "imputation")
        )
        or config["model"] == "smooth"
    )

    plot_losses = config["plot_loss"] and config["task"] == "regression"
    need_attn_weights=(config["model"] == "smooth" or config["model"] == "climax_smooth" or config["model"] == "convit_smooth") and config["smooth_attention"]
    use_smoothing = need_attn_weights and config["task"] == "regression"
    smoothing_lambda = config["reg_lambda"]

    if config["test_only"] == "testset":  # Only evaluate and skip training
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices)

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=model.max_len),
        )
        test_evaluator = runner_class(
            model,
            test_loader,
            device,
            loss_module,
            print_interval=config["print_interval"],
            console=config["console"],
        )
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True, require_padding=require_padding, need_attn_weights=need_attn_weights)
        print_str = "Test Summary: "
        for k, v in aggr_metrics_test.items():
            if k is not None and v is not None:
                print_str += "{}: {:8f} | ".format(k, v)
            else:
                # TODO: pretty print the loss here
                print(aggr_metrics_test)
        logger.info(print_str)
        return

    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(val_data, val_indices)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=model.max_len),
    )

    train_dataset = dataset_class(my_data, train_indices)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=model.max_len),
    )

    trainer = runner_class(
        model,
        train_loader,
        device,
        loss_module,
        optimizer,
        l2_reg=output_reg,
        print_interval=config["print_interval"],
        console=config["console"],
    )
    val_evaluator = runner_class(
        model,
        val_loader,
        device,
        loss_module,
        print_interval=config["print_interval"],
        console=config["console"],
    )

    tensorboard_writer = SummaryWriter(config["tensorboard_dir"])

    # initialize with +inf or -inf depending on key metric
    best_value = 1e16 if config["key_metric"] in NEG_METRICS else -1e16
    # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    metrics = []
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(
        val_evaluator,
        tensorboard_writer,
        config,
        best_metrics,
        best_value,
        epoch=0,
        require_padding=require_padding,
        need_attn_weights=need_attn_weights
    )
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    logger.info("Starting training...")
    train_epochs = []
    train_rmses = []
    val_epochs = []
    val_rmses = []
    all_val_preds = []


    supervised_losses = []
    supervised_smoothness_losses = []

    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc="Training Epoch", leave=False):
        mark = epoch if config["save_all"] else "last"
        epoch_start_time = time.time()
        # dictionary of aggregate epoch metrics
        if plot_losses:
            aggr_metrics_train, _, _, supervised_loss, supervised_smoothness_loss = trainer.train_epoch(config, epoch, keep_predictions=plot_losses, require_padding=require_padding, use_smoothing=use_smoothing, smoothing_lambda=smoothing_lambda, need_attn_weights=need_attn_weights)
            train_epochs.append(epoch)
            train_rmses.append(aggr_metrics_train["loss"] ** 0.5)
            supervised_losses.append(supervised_loss)
            supervised_smoothness_losses.append(supervised_smoothness_loss)
        else:
            aggr_metrics_train = trainer.train_epoch(config, epoch, require_padding=require_padding, use_smoothing=use_smoothing, smoothing_lambda=smoothing_lambda, need_attn_weights=need_attn_weights)
            train_rmses.append(aggr_metrics_train["loss"] ** 0.5)

        if config["baseline"] is not None:
            # early prediction
            if epoch == config["epochs"]:
                aggr_metrics_train, predictions = trainer.train_epoch(config, epoch, keep_predictions=True, need_attn_weights=need_attn_weights)
                aggr_metrics_train, predictions, targets = trainer.train_epoch(config, epoch, keep_predictions=True, need_attn_weights=need_attn_weights)

                example = 1
                dimension = 1
                time_to_forecast = 50
                predictions = predictions[0].cpu().detach().numpy()
                targets = targets[0].cpu().detach().numpy()
                predictions = predictions[example, :, dimension]
                targets = targets[example, :, dimension]

                plt.plot(predictions, label="Predictions", marker="o")
                plt.plot(targets, label="Targets", marker="o")
                plt.legend(["Predictions", "Targets"])
                plt.ylabel("Feature values")
                plt.xlabel("Time step")
                plt.title("Forecast accuracy of autoregressive transformers")
                plt.savefig(os.path.join(config["plot_dir"], "forecast_accuracy.png"))
                plt.close()

        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = "Epoch {} Training Summary: ".format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar("{}/train".format(k), v, epoch)
            print_str += "{}: {:8f} | ".format(k, v)
        logger.info(print_str)
        logger.info(
            "Epoch runtime: {} hours, {} minutes, {} seconds\n".format(
                *utils.readable_time(epoch_runtime)
            )
        )
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info(
            "Avg epoch train. time: {} hours, {} minutes, {} seconds".format(
                *utils.readable_time(avg_epoch_time)
            )
        )
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

        # evaluate if first or last epoch or at specified interval
        if ((epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config["val_interval"] == 0)):
            aggr_metrics_val, best_metrics, best_value = validate(
                val_evaluator,
                tensorboard_writer,
                config,
                best_metrics,
                best_value,
                epoch,
                require_padding=require_padding,
                # keep_predictions=True,
                need_attn_weights=need_attn_weights,
                plot=(epoch != start_epoch + 1) and (epoch != config["epochs"]) and config["plot_accuracy"]
            )
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))
            val_epochs.append(epoch)
            val_rmses.append(aggr_metrics_val['loss'] ** 0.5)
            # all_val_preds.append(predictions)

        utils.save_model(
            os.path.join(config["save_dir"], "model_{}.pth".format(mark)),
            epoch,
            model,
            optimizer,
        )

        # Learning rate scheduling
        if epoch == config["lr_step"][lr_step]:
            utils.save_model(
                os.path.join(config["save_dir"], "model_{}.pth".format(epoch)),
                epoch,
                model,
                optimizer,
            )
            lr = lr * config["lr_factor"][lr_step]
            # so that this index does not get out of bounds
            if lr_step < len(config["lr_step"]) - 1:
                lr_step += 1
            logger.info("Learning rate updated to: ", lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Difficulty scheduling
        if config["harden"] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # Plot training RMSEs
    if config["plot_accuracy"]:
        plt.plot(train_epochs, train_rmses, label="Train")
        plt.plot(val_epochs, val_rmses, label="Val")
        plt.ylabel("RMSE")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Training RMSEs per epoch")
        plt.savefig(os.path.join(config["plot_dir"], "rmses.png"))
        plt.close()

    if plot_losses:
        dataset = config["data_dir"][5:]
        plt.plot(supervised_losses)
        plt.plot(supervised_smoothness_losses)
        plt.legend(["Supervised loss", "Supervised and smoothness loss"])
        plt.ylabel("RMSE")
        plt.xlabel("Epoch")
        plt.title("Training RMSEs per epoch")
        plt.savefig(os.path.join(config["plot_dir"], "supervised_loss.png"))
        plt.close()

        # predictions, attn_weights_layers = self.model(X.to(self.device), padding_masks)

    # # Evaluate each epoch's predictions, and the average prediction
    # all_val_preds = torch.from_numpy(np.stack(all_val_preds, axis=0))  # [epoch, val_examples]
    # all_val_preds = all_val_preds[all_val_preds.shape[0]//2:, :]  # get only the later part of trained models
    # targets = torch.from_numpy(targets.flatten())  # [val_examples]
    # ensemble_preds = torch.mean(all_val_preds, dim=0)
    # epoch_losses = []
    # for i in range(all_val_preds.shape[0]):
    #     epoch_losses.append(loss_module(all_val_preds[i], targets).mean().item())
    # print("===============  Ensemble test =================")
    # print("Epoch losses", epoch_losses)
    # print("Best epoch loss", min(epoch_losses))
    # print("Ensembled pred loss", loss_module(ensemble_preds, targets).mean().item())
    # print("=============================================")

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(
        config["output_dir"], "metrics_" + config["experiment_name"] + ".xls"
    )
    book = utils.export_performance_metrics(
        metrics_filepath, metrics, header, sheet_name="metrics"
    )

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(
        config["records_file"],
        config["initial_timestamp"],
        config["experiment_name"],
        best_metrics,
        aggr_metrics_val,
        comment=config["comment"] + ".  COMMMAND: " + " ".join(sys.argv),
    )

    logger.info(
        "Best {} was {}. Other metrics: {}".format(
            config["key_metric"], best_value, best_metrics
        )
    )
    logger.info("All Done!")

    total_runtime = time.time() - total_start_time
    logger.info(
        "Total runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(total_runtime)
        )
    )

    return best_value


if __name__ == "__main__":
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
