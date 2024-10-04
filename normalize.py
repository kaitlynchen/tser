import pandas as pd
import numpy as np

if __name__ == "__main__":
  PATH = 'climax_appliances_original_hyperparameters_test_set.xls'
  results_df = pd.read_excel(PATH)
  rmses = np.sqrt(results_df["Best loss"])

  # patch_length = [4, 8, 12, 16, 20]  
  # stride = [0.25, 0.5, 0.75, 1]
  # encoder_layers = [1, 3, 5, 7]
  # smooth_factor = [1e-3, 1e-2, 1e-1]
  # heads = [2, 4, 8, 16]
  # d_model = [128, 144]
  # d_model = [64, 128, 144, 256, 512]
  # batch = [64, 128, 256, 512]
  # dim_ffw = [64, 128, 256, 512]
  smooth_lambda = [0.001, 0.0003, 0.0001, 0.00003, 0.00001]

  print("RMSEs: ", rmses)
  print("Mean loss:", np.mean(rmses))
  print("Standard deviation of loss:", np.std(rmses))

  # for i, lr in enumerate(smooth_lambda):
  #     print("Mean loss of smoothing_lambda", lr, ":", np.mean(rmses.iloc[i::5]))
  #     print("Stdev loss of smoothing_lambda", lr, ":", np.std(rmses.iloc[i::5]))

  # for i, lr in enumerate(heads):
  #   for j, layer in enumerate(d_model):
  #     print("Mean loss of head", lr, "and d_model", layer, ":", np.mean(rmses.iloc[(i * len(d_model) + j)::(len(heads) * len(d_model))]))

  # for i, lr in enumerate(patch_length):
  #   for j, layer in enumerate(stride):
  #     print("Mean loss of patch length ", lr, " and stride ", layer, ": ", np.mean(rmses[(i * len(stride) + j)::(len(patch_length) * len(stride))]))
      # print("Standard deviation of patch length ", lr, " and stride ", layer, ": ", np.std(rmses[(i * len(stride) + j)::(len(patch_length) * len(stride))]))

  # for i, lr in enumerate(encoder_layers):
  #   for j, layer in enumerate(smooth_factor):
  #     print("Mean loss of learning rate ", lr, " and ", layer, " gpsa layers: ", np.mean(rmses[(i * len(smooth_factor) + j)::(len(encoder_layers) * len(smooth_factor))]))
      # print("Standard deviation of learning rate ", lr, " and ", layer, " gpsa layers: ", np.std(rmses[(i * len(smooth_factor) + j)::(len(encoder_layers) * len(smooth_factor))]))

  # seed = [0, 1, 2, 3, 4]
  # patch = [1, 4, 8, 16]
  # stride_factor = [0.5, 1]
  # learning_rate = [1e-3, 1e-2, 1e-1]
  # smoothing_factor = [0, 1e-3, 1e-2, 1e-1]

  # results_df.insert(8, "RMSE", rmses, True)

  # for i, patch_size in enumerate(patch):
  #   for j, stride in enumerate(stride_factor):
  #     for k, lr in enumerate(learning_rate):
  #       for l, smooth in enumerate(smoothing_factor):
  #         print("Mean loss of patch length ", patch_size, ", stride ", stride, ", learning rate ", lr, " and smoothing factor ", smooth, ": ", np.mean(rmses[(i * len(stride_factor) + j * len(learning_rate) + k * len(smoothing_factor) + l)::(len(patch) * len(stride_factor) * len(learning_rate) * len(smoothing_factor) - len(learning_rate) * len(smoothing_factor))]))

