# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 01:05:24 2021

@author: Ranak Roy Chowdhury
"""
import warnings
import pickle
import torch
import math
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import multitask_transformer_class
warnings.filterwarnings("ignore")

# The following code is adapted from the python package sktime to read .ts file.


class TsFileParseException(Exception):
    """
    Should be raised when parsing a .ts file and the format is incorrect.
    """
    pass


def load_from_tsfile_to_dataframe(full_file_path_and_name, return_separate_X_and_y=True,
                                  replace_missing_vals_with='NaN'):
    """Loads data from a .ts file into a Pandas DataFrame.
    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced with prior to parsing.
    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a numpy array containing the relevant time-series and corresponding class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing all time-series and (if relevant) a column "class_vals" the associated class values.
    """

    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_target_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_float = None
    previous_timestamp_was_int = None
    previous_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0

    # Parse the file
    # print(full_file_path_and_name)
    with open(full_file_path_and_name, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            # print(".", end='')
            # Strip white space from start/end of line and change to lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException(
                            "metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "problemname tag requires an associated value")

                    problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException(
                            "metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len != 2:
                        raise TsFileParseException(
                            "timestamps tag requires an associated Boolean value")
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise TsFileParseException("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException(
                            "metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)
                    if token_len != 2:
                        raise TsFileParseException(
                            "univariate tag requires an associated Boolean value")
                    elif tokens[1] == "true":
                        univariate = True
                    elif tokens[1] == "false":
                        univariate = False
                    else:
                        raise TsFileParseException("invalid univariate value")

                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException(
                            "metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "classlabel tag requires an associated Boolean value")

                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise TsFileParseException("invalid classLabel value")

                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise TsFileParseException(
                            "if the classlabel tag is true then class values must be supplied")

                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                elif line.startswith("@targetlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException(
                            "metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "targetlabel tag requires an associated Boolean value")

                    if tokens[1] == "true":
                        target_labels = True
                    elif tokens[1] == "false":
                        target_labels = False
                    else:
                        raise TsFileParseException("invalid targetLabel value")

                    has_target_labels_tag = True
                    class_val_list = []
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise TsFileParseException(
                            "data tag should not have an associated value")

                    if data_started and not metadata_started:
                        raise TsFileParseException(
                            "metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    incomplete_regression_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_target_labels_tag or not has_data_tag
                    incomplete_classification_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_class_labels_tag or not has_data_tag
                    if incomplete_regression_meta_data and incomplete_classification_meta_data:
                        raise TsFileParseException(
                            "a full set of metadata has not been provided before the data")

                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)

                    # Check if we dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False

                        timestamps_for_dimension = []
                        values_for_dimension = []

                        this_line_num_dimensions = 0
                        line_len = len(line)
                        char_num = 0

                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1

                            # See if there is any more data to read in or if we should validate that read thus far

                            if char_num < line_len:

                                # See if we have an empty dimension (i.e. no values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dimensions + 1):
                                        instance_list.append([])

                                    instance_list[this_line_num_dimensions].append(
                                        pd.Series())
                                    this_line_num_dimensions += 1

                                    has_another_value = False
                                    has_another_dimension = True

                                    timestamps_for_dimension = []
                                    values_for_dimension = []

                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and target_labels:
                                        class_val = line[char_num:].strip()

                                        # if class_val not in class_val_list:
                                        #     raise TsFileParseException(
                                        #         "the class value '" + class_val + "' on line " + str(
                                        #             line_num + 1) + " is not valid")

                                        class_val_list.append(float(class_val))
                                        char_num = line_len

                                        has_another_value = False
                                        has_another_dimension = False

                                        timestamps_for_dimension = []
                                        values_for_dimension = []

                                    else:

                                        # Read in the data contained within the next tuple

                                        if line[char_num] != "(" and not target_labels:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " does not start with a '('")

                                        char_num += 1
                                        tuple_data = ""

                                        while char_num < line_len and line[char_num] != ")":
                                            tuple_data += line[char_num]
                                            char_num += 1

                                        if char_num >= line_len or line[char_num] != ")":
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " does not end with a ')'")

                                        # Read in any spaces immediately after the current tuple

                                        char_num += 1

                                        while char_num < line_len and str.isspace(line[char_num]):
                                            char_num += 1

                                        # Check if there is another value or dimension to process after this tuple

                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False

                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False

                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True

                                        char_num += 1

                                        # Get the numeric value for the tuple by reading from the end of the tuple data backwards to the last comma

                                        last_comma_index = tuple_data.rfind(
                                            ',')

                                        if last_comma_index == -1:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains a tuple that has no comma inside of it")

                                        try:
                                            value = tuple_data[last_comma_index + 1:]
                                            value = float(value)

                                        except ValueError:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains a tuple that does not have a valid numeric value")

                                        # Check the type of timestamp that we have

                                        timestamp = tuple_data[0: last_comma_index]

                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False

                                        if not timestamp_is_int:
                                            try:
                                                timestamp = float(timestamp)
                                                timestamp_is_float = True
                                                timestamp_is_timestamp = False
                                            except ValueError:
                                                timestamp_is_float = False

                                        if not timestamp_is_int and not timestamp_is_float:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False

                                        # Make sure that the timestamps in the file (not just this dimension or case) are consistent

                                        if not timestamp_is_timestamp and not timestamp_is_int and not timestamp_is_float:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains a tuple that has an invalid timestamp '" + timestamp + "'")

                                        if previous_timestamp_was_float is not None and previous_timestamp_was_float and not timestamp_is_float:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                        if previous_timestamp_was_int is not None and previous_timestamp_was_int and not timestamp_is_int:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                        if previous_timestamp_was_timestamp is not None and previous_timestamp_was_timestamp and not timestamp_is_timestamp:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                        # Store the values

                                        timestamps_for_dimension += [timestamp]
                                        values_for_dimension += [value]

                                        #  If this was our first tuple then we store the type of timestamp we had

                                        if previous_timestamp_was_timestamp is None and timestamp_is_timestamp:
                                            previous_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False
                                            previous_timestamp_was_float = False

                                        if previous_timestamp_was_int is None and timestamp_is_int:
                                            previous_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                            previous_timestamp_was_float = False

                                        if previous_timestamp_was_float is None and timestamp_is_float:
                                            previous_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = False
                                            previous_timestamp_was_float = True

                                        # See if we should add the data for this dimension

                                        if not has_another_value:
                                            if len(instance_list) < (this_line_num_dimensions + 1):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamps_for_dimension = pd.DatetimeIndex(
                                                    timestamps_for_dimension)

                                            instance_list[this_line_num_dimensions].append(
                                                pd.Series(index=timestamps_for_dimension, data=values_for_dimension))
                                            this_line_num_dimensions += 1

                                            timestamps_for_dimension = []
                                            values_for_dimension = []

                            elif has_another_value:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ',' that is not followed by another tuple")

                            elif has_another_dimension and target_labels:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ':' while it should list a class value")

                            elif has_another_dimension and not target_labels:
                                if len(instance_list) < (this_line_num_dimensions + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dimensions].append(
                                    pd.Series(dtype=np.float32))
                                this_line_num_dimensions += 1
                                num_dimensions = this_line_num_dimensions

                            # If this is the 1st line of data we have seen then note the dimensions

                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dimensions

                                if num_dimensions != this_line_num_dimensions:
                                    raise TsFileParseException("line " + str(
                                        line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                        # Check that we are not expecting some more data, and if not, store that processed above

                        if has_another_value:
                            raise TsFileParseException(
                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                    line_num + 1) + " ends with a ',' that is not followed by another tuple")

                        elif has_another_dimension and target_labels:
                            raise TsFileParseException(
                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                    line_num + 1) + " ends with a ':' while it should list a class value")

                        elif has_another_dimension and not target_labels:
                            if len(instance_list) < (this_line_num_dimensions + 1):
                                instance_list.append([])

                            instance_list[this_line_num_dimensions].append(
                                pd.Series())
                            this_line_num_dimensions += 1
                            num_dimensions = this_line_num_dimensions

                        # If this is the 1st line of data we have seen then note the dimensions

                        if not has_another_value and num_dimensions != this_line_num_dimensions:
                            raise TsFileParseException("line " + str(
                                line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                        # Check if we should have class values, and if so that they are contained in those listed in the metadata

                        if target_labels and len(class_val_list) == 0:
                            raise TsFileParseException(
                                "the cases have no associated class values")
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)

                            if target_labels:
                                num_dimensions -= 1

                            for dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False

                        # See how many dimensions that the case whose data in represented in this line has
                        this_line_num_dimensions = len(dimensions)

                        if target_labels:
                            this_line_num_dimensions -= 1

                        # All dimensions should be included for all series, even if they are empty
                        if this_line_num_dimensions != num_dimensions:
                            raise TsFileParseException("inconsistent number of dimensions. Expecting " + str(
                                num_dimensions) + " but have read " + str(this_line_num_dimensions))

                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(
                                    pd.Series(data_series))
                            else:
                                instance_list[dim].append(pd.Series())

                        if target_labels:
                            class_val_list.append(
                                float(dimensions[num_dimensions].strip()))

            line_num += 1

    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        complete_regression_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_target_labels_tag and has_data_tag
        complete_classification_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_class_labels_tag and has_data_tag

        if metadata_started and not complete_regression_meta_data and not complete_classification_meta_data:
            raise TsFileParseException("metadata incomplete")
        elif metadata_started and not data_started:
            raise TsFileParseException("file contained metadata but no data")
        elif metadata_started and data_started and len(instance_list) == 0:
            raise TsFileParseException("file contained metadata but no data")

        # Create a DataFrame from the data parsed above
        data = pd.DataFrame(dtype=np.float32)

        for dim in range(0, num_dimensions):
            data['dim_' + str(dim)] = instance_list[dim]

        # Check if we should return any associated class labels separately

        if target_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data['class_vals'] = pd.Series(class_val_list)
                return data
        else:
            return data
    else:
        raise TsFileParseException("empty file")

# loading optimized hyperparameters


def get_optimized_hyperparameters(dataset):

    path = './hyperparameters.pkl'
    with open(path, 'rb') as handle:
        all_datasets = pickle.load(handle)
        if dataset in all_datasets:
            prop = all_datasets[dataset]
    return prop


# loading user-specified hyperparameters
def get_user_specified_hyperparameters(args):

    prop = {}
    prop['batch'], prop['lr'], prop['nlayers'], prop['emb_size'], prop['nhead'], prop['task_rate'], prop['masking_ratio'], prop['task_type'] = \
        args.batch, args.lr, args.nlayers, args.emb_size, args.nhead, args.task_rate, args.masking_ratio, args.task_type
    return prop


# loading fixed hyperparameters
def get_fixed_hyperparameters(prop, args):

    prop['lamb'], prop['epochs'], prop['ratio_highest_attention'], prop['avg'] = args.lamb, args.epochs, args.ratio_highest_attention, args.avg
    prop['dropout'], prop['nhid'], prop['nhid_task'], prop['nhid_tar'], prop[
        'dataset'] = args.dropout, args.nhid, args.nhid_task, args.nhid_tar, args.dataset
    return prop


def get_prop(args):

    # loading optimized hyperparameters
    # prop = get_optimized_hyperparameters(args.dataset)

    # loading user-specified hyperparameters
    prop = get_user_specified_hyperparameters(args)

    # loading fixed hyperparameters
    prop = get_fixed_hyperparameters(prop, args)
    return prop


def data_loader(dataset, data_path, task_type):
    X_train = np.load(os.path.join(data_path + 'X_train.npy'),
                      allow_pickle=True).astype(np.float)
    X_test = np.load(os.path.join(data_path + 'X_test.npy'),
                     allow_pickle=True).astype(np.float)

    if task_type == 'classification':
        y_train = np.load(os.path.join(
            data_path + 'y_train.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(
            data_path + 'y_test.npy'), allow_pickle=True)
    else:
        y_train = np.load(os.path.join(data_path + 'y_train.npy'),
                          allow_pickle=True).astype(np.float)
        y_test = np.load(os.path.join(data_path + 'y_test.npy'),
                         allow_pickle=True).astype(np.float)

    return X_train, y_train, X_test, y_test


def make_perfect_batch(X, num_inst, num_samples):
    extension = np.zeros((num_samples - num_inst, X.shape[1], X.shape[2]))
    X = np.concatenate((X, extension), axis=0)
    return X


def mean_standardize_fit(X):
    m1 = np.mean(X, axis=1)
    mean = np.mean(m1, axis=0)

    s1 = np.std(X, axis=1)
    std = np.mean(s1, axis=0)

    return mean, std


def mean_standardize_transform(X, mean, std):
    return (X - mean) / std


def uniform_scaling(data, max_len):
    """
    This is a function to scale the time series uniformly
    :param data:
    :param max_len:
    :return:
    """
    seq_len = len(data)
    scaled_data = [data[int(j * seq_len / max_len)] for j in range(max_len)]

    return scaled_data


def process_data(X, min_len, normalise=None):
    """
    This is a function to process the data, i.e. convert dataframe to numpy array
    :param X:
    :param min_len:
    :param normalise:
    :return:
    """
    tmp = []
    for i in tqdm(range(len(X))):
        _x = X.iloc[i, :].copy(deep=True)

        # 1. find the maximum length of each dimension
        all_len = [len(y) for y in _x]
        max_len = max(all_len)

        # 2. adjust the length of each dimension
        _y = []
        for y in _x:
            # 2.1 fill missing values
            if y.isnull().any():
                y = y.interpolate(method='linear', limit_direction='both')

            # 2.2. if length of each dimension is different, uniformly scale the shorted one to the max length
            if len(y) < max_len:
                y = uniform_scaling(y, max_len)
            _y.append(y)
        _y = np.array(np.transpose(_y))

        # 3. adjust the length of the series, chop of the longer series
        _y = _y[:min_len, :]

        # 4. normalise the series
        if normalise == "standard":
            scaler = StandardScaler().fit(_y)
            _y = scaler.transform(_y)
        if normalise == "minmax":
            scaler = MinMaxScaler().fit(_y)
            _y = scaler.transform(_y)

        tmp.append(_y)
    X = np.array(tmp)
    return X


def preprocess(prop, X_train, y_train, X_test, y_test):
    mean, std = mean_standardize_fit(X_train)
    X_train, X_test = mean_standardize_transform(
        X_train, mean, std), mean_standardize_transform(X_test, mean, std)

    num_train_inst, num_test_inst = X_train.shape[0], X_test.shape[0]
    num_train_samples = math.ceil(
        num_train_inst / prop['batch']) * prop['batch']
    num_test_samples = math.ceil(num_test_inst / prop['batch']) * prop['batch']

    X_train = make_perfect_batch(X_train, num_train_inst, num_train_samples)
    X_test = make_perfect_batch(X_test, num_test_inst, num_test_samples)

    X_train_task = torch.as_tensor(X_train).float()
    X_test = torch.as_tensor(X_test).float()

    if prop['task_type'] == 'classification':
        y_train_task = torch.as_tensor(y_train)
        y_test = torch.as_tensor(y_test)
    else:
        y_train_task = torch.as_tensor(y_train).float()
        y_test = torch.as_tensor(y_test).float()

    return X_train_task, y_train_task, X_test, y_test


def initialize_training(prop):
    model = multitask_transformer_class.MultitaskTransformerModel(prop['task_type'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'],
                                                                  prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nhid_tar'], prop['nhid_task'], prop['nlayers'], prop['dropout']).to(prop['device'])
    best_model = multitask_transformer_class.MultitaskTransformerModel(prop['task_type'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'],
                                                                       prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nhid_tar'], prop['nhid_task'], prop['nlayers'], prop['dropout']).to(prop['device'])

    criterion_tar = torch.nn.MSELoss()
    criterion_task = torch.nn.CrossEntropyLoss(
    ) if prop['task_type'] == 'classification' else torch.nn.MSELoss()  # nn.L1Loss() for MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=prop['lr'])
    best_optimizer = torch.optim.Adam(
        best_model.parameters(), lr=prop['lr'])  # get new optimiser

    return model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer


def attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights):
    # attention_weights = attention_weights.to('cpu')
    # instance_weights = torch.sum(attention_weights, axis = 1)
    res, index = instance_weights.topk(
        int(math.ceil(ratio_highest_attention * X.shape[1])))
    index = index.cpu().data.tolist()
    index2 = [random.sample(index[i], int(
        math.ceil(masking_ratio * X.shape[1]))) for i in range(X.shape[0])]
    return np.array(index2)


def random_instance_masking(X, masking_ratio, ratio_highest_attention, instance_weights):
    indices = attention_sampled_masking_heuristic(
        X, masking_ratio, ratio_highest_attention, instance_weights)
    boolean_indices = np.array(
        [[True if i in index else False for i in range(X.shape[1])] for index in indices])
    boolean_indices_masked = np.repeat(
        boolean_indices[:, :, np.newaxis], X.shape[2], axis=2)
    boolean_indices_unmasked = np.invert(boolean_indices_masked)

    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = np.copy(
        X), np.copy(X), np.copy(X)
    X_train_tar = np.where(boolean_indices_unmasked, X, 0.0)
    y_train_tar_masked = y_train_tar_masked[boolean_indices_masked].reshape(
        X.shape[0], -1)
    y_train_tar_unmasked = y_train_tar_unmasked[boolean_indices_unmasked].reshape(
        X.shape[0], -1)
    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = torch.as_tensor(X_train_tar).float(
    ), torch.as_tensor(y_train_tar_masked).float(), torch.as_tensor(y_train_tar_unmasked).float()

    return X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked


def compute_tar_loss(model, device, criterion_tar, y_train_tar_masked, y_train_tar_unmasked, batched_input_tar,
                     batched_boolean_indices_masked, batched_boolean_indices_unmasked, num_inst, start):
    model.train()
    out_tar = model(torch.as_tensor(batched_input_tar,
                    device=device), 'reconstruction')[0]

    out_tar_masked = torch.as_tensor(out_tar[torch.as_tensor(
        batched_boolean_indices_masked)].reshape(out_tar.shape[0], -1), device=device)
    out_tar_unmasked = torch.as_tensor(out_tar[torch.as_tensor(
        batched_boolean_indices_unmasked)].reshape(out_tar.shape[0], -1), device=device)

    loss_tar_masked = criterion_tar(out_tar_masked[: num_inst], torch.as_tensor(
        y_train_tar_masked[start: start + num_inst], device=device))
    loss_tar_unmasked = criterion_tar(out_tar_unmasked[: num_inst], torch.as_tensor(
        y_train_tar_unmasked[start: start + num_inst], device=device))

    return loss_tar_masked, loss_tar_unmasked


def compute_task_loss(nclasses, model, device, criterion_task, y_train_task, batched_input_task, task_type, num_inst, start):
    model.train()
    out_task, attn = model(torch.as_tensor(
        batched_input_task, device=device), task_type)
    out_task = out_task.view(-1,
                             nclasses) if task_type == 'classification' else out_task.squeeze()
    loss_task = criterion_task(out_task[: num_inst], torch.as_tensor(
        y_train_task[start: start + num_inst], device=device))  # dtype = torch.long
    return attn, loss_task


def multitask_train(model, criterion_tar, criterion_task, optimizer, X_train_tar, X_train_task, y_train_tar_masked, y_train_tar_unmasked,
                    y_train_task, boolean_indices_masked, boolean_indices_unmasked, prop):

    model.train()  # Turn on the train mode
    total_loss_tar_masked, total_loss_tar_unmasked, total_loss_task = 0.0, 0.0, 0.0
    num_batches = math.ceil(X_train_tar.shape[0] / prop['batch'])
    output, attn_arr = [], []

    for i in range(num_batches):
        start = int(i * prop['batch'])
        end = int((i + 1) * prop['batch'])
        num_inst = y_train_task[start: end].shape[0]

        optimizer.zero_grad()

        batched_input_tar = X_train_tar[start: end]
        batched_input_task = X_train_task[start: end]
        batched_boolean_indices_masked = boolean_indices_masked[start: end]
        batched_boolean_indices_unmasked = boolean_indices_unmasked[start: end]

        loss_tar_masked, loss_tar_unmasked = compute_tar_loss(model, prop['device'], criterion_tar, y_train_tar_masked, y_train_tar_unmasked,
                                                              batched_input_tar, batched_boolean_indices_masked, batched_boolean_indices_unmasked, num_inst, start)

        attn, loss_task = compute_task_loss(prop['nclasses'], model, prop['device'], criterion_task, y_train_task,
                                            batched_input_task, prop['task_type'], num_inst, start)

        total_loss_tar_masked += loss_tar_masked.item()
        total_loss_tar_unmasked += loss_tar_unmasked.item()
        total_loss_task += loss_task.item() * num_inst

        # a = list(train_model.parameters())[0].clone()
        loss = prop['task_rate'] * (prop['lamb'] * loss_tar_masked + (
            1 - prop['lamb']) * loss_tar_unmasked) + (1 - prop['task_rate']) * loss_task
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # b = list(train_model.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))

        # if list(model.parameters())[0].grad is None:
        #    print("None")

        # remove the diagonal values of the attention map while aggregating the column wise attention scores
        attn_arr.append(torch.sum(attn, axis=1) -
                        torch.diagonal(attn, offset=0, dim1=1, dim2=2))

    instance_weights = torch.cat(attn_arr, axis=0)
    return total_loss_tar_masked, total_loss_tar_unmasked, total_loss_task / y_train_task.shape[0], instance_weights


def evaluate(y_pred, y, nclasses, criterion, task_type, device, avg):
    results = []

    if task_type == 'classification':
        loss = criterion(y_pred.view(-1, nclasses),
                         torch.as_tensor(y, device=device)).item()

        pred, target = y_pred.cpu().data.numpy(), y.cpu().data.numpy()
        pred = np.argmax(pred, axis=1)
        acc = accuracy_score(target, pred)
        prec = precision_score(target, pred, average=avg)
        rec = recall_score(target, pred, average=avg)
        f1 = f1_score(target, pred, average=avg)

        results.extend([loss, acc, prec, rec, f1])
    else:
        y_pred = y_pred.squeeze()
        y = torch.as_tensor(y, device=device)
        rmse = math.sqrt(((y_pred - y) * (y_pred - y)
                          ).sum().data / y_pred.shape[0])
        mae = (torch.abs(y_pred - y).sum().data / y_pred.shape[0]).item()
        results.extend([rmse, mae])
    # per_class_results = precision_recall_fscore_support(target, pred, average = None, labels = list(range(0, nclasses)))

    return results


def test(model, X, y, batch, nclasses, criterion, task_type, device, avg):
    model.eval()  # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)

    output_arr = []
    with torch.no_grad():
        for i in range(num_batches):
            start = int(i * batch)
            end = int((i + 1) * batch)
            num_inst = y[start: end].shape[0]

            out = model(torch.as_tensor(
                X[start: end], device=device), task_type)[0]
            output_arr.append(out[: num_inst])

    return evaluate(torch.cat(output_arr, 0), y, nclasses, criterion, task_type, device, avg)


def training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, X_test, y_test, prop):
    tar_loss_masked_arr, tar_loss_unmasked_arr, tar_loss_arr, task_loss_arr, min_task_loss = [
    ], [], [], [], math.inf
    acc, rmse, mae = 0, math.inf, math.inf

    instance_weights = torch.as_tensor(torch.rand(
        X_train_task.shape[0], prop['seq_len']), device=prop['device'])
    for epoch in range(1, prop['epochs'] + 1):

        X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked = \
            random_instance_masking(
                X_train_task, prop['masking_ratio'], prop['ratio_highest_attention'], instance_weights)

        tar_loss_masked, tar_loss_unmasked, task_loss, instance_weights = multitask_train(model, criterion_tar, criterion_task, optimizer,
                                                                                          X_train_tar, X_train_task, y_train_tar_masked, y_train_tar_unmasked, y_train_task,
                                                                                          boolean_indices_masked, boolean_indices_unmasked, prop)

        tar_loss_masked_arr.append(tar_loss_masked)
        tar_loss_unmasked_arr.append(tar_loss_unmasked)
        tar_loss = tar_loss_masked + tar_loss_unmasked
        tar_loss_arr.append(tar_loss)
        task_loss_arr.append(task_loss)
        print('Epoch: ' + str(epoch) + ', TAR Loss: ' +
              str(tar_loss), ', TASK Loss: ' + str(task_loss))

        # save model and optimizer for lowest training loss on the end task
        if task_loss < min_task_loss:
            min_task_loss = task_loss
            best_model.load_state_dict(model.state_dict())
            best_optimizer.load_state_dict(optimizer.state_dict())

        # Saved best model state at the lowest training loss is evaluated on the official test set
        test_metrics = test(best_model, X_test, y_test,
                            prop['batch'], prop['nclasses'], criterion_task, prop['task_type'], prop['device'], prop['avg'])

        if prop['task_type'] == 'classification' and test_metrics[1] > acc:
            acc = test_metrics[1]
        elif prop['task_type'] == 'regression' and test_metrics[0] < rmse:
            rmse = test_metrics[0]
            mae = test_metrics[1]

    if prop['task_type'] == 'classification':
        print('Dataset: ' + prop['dataset'] + ', Acc: ' + str(acc))
    elif prop['task_type'] == 'regression':
        print('Dataset: ' + prop['dataset'] +
              ', RMSE: ' + str(rmse) + ', MAE: ' + str(mae))

    del model
    torch.cuda.empty_cache()
