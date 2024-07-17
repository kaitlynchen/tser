"""
Simple linaer baseline
"""

import random

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from optimizers import get_optimizer
from models.loss import get_loss_module
from datasets.datasplit import split_dataset
from datasets.data import data_factory, Normalizer
from datasets.utils import process_data
from utils import utils, visualization_utils
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
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required due to CUBLAS https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.use_deterministic_algorithms(True)
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
    test_data = None
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
        raise ValueError("Validation set not supported for linear.")
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
        raise ValueError("Validation set not supported for linear.")
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

    print("Train data", my_data.feature_df.loc[train_indices].shape, my_data.labels_df.loc[train_indices].shape)
    print("Test data", test_data.feature_df.loc[test_indices].shape, test_data.labels_df.loc[test_indices].shape)

    X_train = np.empty((len(train_indices), my_data.max_seq_len, my_data.feature_df.shape[1]))
    for i, train_idx in enumerate(train_indices):
        X_train[i] = my_data.feature_df.loc[train_idx]  # RHS is [timesteps, variables]
    X_test = np.empty((len(test_indices), my_data.max_seq_len, my_data.feature_df.shape[1]))
    for i, test_idx in enumerate(test_indices):
        X_test[i] = test_data.feature_df.loc[test_idx]
    Y_train = my_data.labels_df.loc[train_indices].to_numpy()
    Y_test = test_data.labels_df.loc[test_indices].to_numpy()
    print("Shape test", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)  # X: [n_examples, n_timsteps, n_vars]  Y: [n_examples]

    # Flatten
    #X_train = X_train.mean(axis=1)
    #X_test = X_test.mean(axis=1)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()
    print("Shape test after flatten", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


    # Train model
    if args.model == "ridge":
        model = Ridge(random_state=args.seed)
        hyperparams = {"alpha": [0, 0.01, 0.1, 1, 10, 100, 1000]}
    elif args.model == "lasso":
        model = Lasso(random_state=args.seed)
        hyperparams = {"alpha": [0, 0.01, 0.1, 1, 10, 100, 1000]}
    regressor = GridSearchCV(model, hyperparams, cv=10)
    regressor = regressor.fit(X_train, Y_train)
    predictions_test = regressor.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(Y_test, predictions_test))
    print("RMSE", rmse_test)
    coefs = regressor.best_estimator_.coef_.reshape((my_data.max_seq_len, my_data.feature_df.shape[1]))
    print("Coefs", coefs.shape, coefs)

    # Visualizations
    visualization_utils.plot_single_scatter_file(predictions_test, Y_test, "predicted", "true", config['plot_dir'],
                                                 title_description=f"{config['model']} flattened",
                                                 filename_description="test", should_align=True)

    # Plot time series
    visualization_utils.plot_time_series(coefs, os.path.join(config['plot_dir'], f"{config['model']}_coefs.png"))
    visualization_utils.plot_time_series(X_train[0].reshape((my_data.max_seq_len, my_data.feature_df.shape[1])),
                                         os.path.join(config['plot_dir'], f"{config['model']}_example_x0.png"))


if __name__ == "__main__":
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
