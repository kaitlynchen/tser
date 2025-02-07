import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from utils.tools import create_directory
from visualize import plot_scatter_labels

module = "RegressionExperiment"
data_path = "data/"
# see data_loader.regression_datasets
problems = ["AppliancesEnergy"]
# see regressor_tools.all_models
regressors = ["ridge"]
iterations = [1]
norm = "standard"               # none, standard, minmax

output_path = "output/regression_standard/AppliancesEnergy"
if __name__ == '__main__':
    # for each problem
    for problem in problems:
        print("#########################################################################")
        print("[{}] Starting Experiments".format(module))
        print("#########################################################################")
        print("[{}] Data path: {}".format(module, data_path))
        print("[{}] Problem: {}".format(module, problem))

        # set data folder, train & test
        data_folder = data_path + problem + "/"
        train_file = data_folder + problem + "_TRAIN.ts"
        test_file = data_folder + problem + "_TEST.ts"

        # loading the data. X_train and X_test are dataframe of N x n_dim
        print("[{}] Loading data".format(module))
        X_train, y_train = load_from_tsfile_to_dataframe(train_file)
        X_test, y_test = load_from_tsfile_to_dataframe(test_file)

        print("[{}] X_train: {}".format(module, X_train.shape))
        print("[{}] X_test: {}".format(module, X_test.shape))

        # in case there are different lengths in the dataset, we need to consider that.
        # assume that all the dimensions are the same length
        print("[{}] Finding minimum length".format(module))
        min_len = np.inf
        for i in range(len(X_train)):
            x = X_train.iloc[i, :]
            all_len = [len(y) for y in x]
            min_len = min(min(all_len), min_len)
        for i in range(len(X_test)):
            x = X_test.iloc[i, :]
            all_len = [len(y) for y in x]
            min_len = min(min(all_len), min_len)
        print("[{}] Minimum length: {}".format(module, min_len))

        # process the data into numpy array
        print("[{}] Reshaping data".format(module))
        x_train = process_data(X_train, normalise=norm, min_len=min_len)
        x_test = process_data(X_test, normalise=norm, min_len=min_len)

        print("[{}] X_train: {}".format(module, x_train.shape))
        print("[{}] X_test: {}".format(module, x_test.shape))

        test_avg = np.mean(x_test, axis=1, dtype=np.float32)
        test_avg = np.expand_dims(test_avg, axis=1)
        print("[{}] X_test average: {}".format(module, test_avg.shape))

        # split data for baseline 1
        proportions = [0.25, 0.5, 0.75, 1]
        rmses = []

        for proportion in proportions:
            print("[{}] Replacing data with historical average".format(module))
            # baseline 1
            train_index = int(x_train.shape[1] * proportion)
            test_index = int(x_test.shape[1] * proportion)
            reshaped_x_train = x_train[:, :train_index, :]
            reshaped_x_test = x_test[:, :test_index, :]

            # baseline 2
            # reshaped_x_train = x_train
            # reshaped_x_test = np.copy(x_test)
            # reshaped_x_test[:, test_index:, :] = test_avg

            print("[{}] X_train: {}".format(module, reshaped_x_train.shape))
            print("[{}] X_test: {}".format(module, reshaped_x_test.shape))

            for regressor_name in regressors:
                print("[{}] Regressor: {}".format(module, regressor_name))
                for itr in iterations:
                    # create output directory
                    output_directory = "output/regression/"
                    if norm != "none":
                        output_directory = "output/regression_{}/".format(norm)
                    output_directory = output_directory + regressor_name + \
                        '/' + problem + '/itr_' + str(itr) + '/'
                    create_directory(output_directory)

                    print("[{}] Iteration: {}".format(module, itr))
                    print("[{}] Output Dir: {}".format(
                        module, output_directory))

                    # fit the regressor
                    regressor = fit_regressor(
                        output_directory, regressor_name, reshaped_x_train, y_train, reshaped_x_test, y_test, itr=itr)

                    # start testing
                    y_pred = regressor.predict(reshaped_x_test)
                    df_metrics = calculate_regression_metrics(y_test, y_pred)

                    print(df_metrics)
                    rmses.append(df_metrics['rmse'])

                    # save the outputs
                    # df_metrics.to_csv(output_directory +
                    #                   'regression_experiment.csv', index=False)]

        print("RMSEs: ", rmses)
