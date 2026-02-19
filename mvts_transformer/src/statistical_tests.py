from scipy.stats import wilcoxon, ttest_ind,  ttest_rel
import numpy as np
import pandas as pd
import os

OUTPUT_DIR = "~/tser/mvts_transformer/src/output"
PAIRED_FILES = {
    "AppliancesEnergy": ["output_zerveas/AppliancesEnergy_REPRO_ZERVEAS.xls", "AppliancesEnergy_LOCAL_per_timestep_TESTBEST.xls",
                         "AppliancesEnergy_LOCAL_lstm_TESTBEST.xls", "AppliancesEnergy_LADAA_TUNING_TESTBEST.xls"],  # "AppliancesEnergy_CONVALIBI15_TUNINGL1_VALTEMPORAL_TESTBEST.xls"],
    "BeijingPM10Quality": ["output_zerveas/BeijingPM10Quality_REPRO_ZERVEAS.xls", "BeijingPM10Quality_LOCAL_per_timestep_TESTBEST.xls", 
                           "BeijingPM10Quality_LOCAL_lstm_TESTBEST.xls", "BeijingPM10Quality_CONVALIBI13_TUNINGL1_TESTBEST.xls"],
    "BeijingPM25Quality": ["output_zerveas/BeijingPM25Quality_REPRO_ZERVEAS.xls", "BeijingPM25Quality_LOCAL_per_timestep_TESTBEST.xls",
                           "BeijingPM25Quality_LOCAL_lstm_TESTBEST.xls", "BeijingPM25Quality_CONVALIBI12_TESTBEST.xls"],
    "BenzeneConcentration": ["output_zerveas/BenzeneConcentration_REPRO_ZERVEAS.xls", "BenzeneConcentration_per_timestep_TUNING_FIXEDSTRIDE_SEQSPLIT2_TESTBEST.xls",  # "BenzeneConcentration_LOCAL_per_timestep_TESTBEST.xls",
                             "BenzeneConcentration_lstm_TUNING_FIXEDSTRIDE_SEQSPLIT2_TESTBEST.xls", #BenzeneConcentration_LOCAL_lstm_TESTBEST.xls",
                             "BenzeneConcentration_CONVALIBI12_TESTBEST.xls"],
    "IEEEPPG": ["output_zerveas/IEEEPPG_REPRO_ZERVEAS.xls", "IEEEPPG_per_timestep_TUNING_FIXEDSTRIDE_SEQSPLIT_TESTBEST.xls",
                "IEEEPPG_lstm_TUNING_FIXEDSTRIDE_SEQSPLIT_TESTBEST.xls", "IEEEPPG_LADAA_TUNING_FIXEDSTRIDE_TESTBEST.xls"],
                
                # "IEEEPPG_LOCAL42_per_timestep_TESTBEST.xls", 
                # "IEEEPPG_LOCAL_lstm_TESTBEST.xls", # "IEEEPPG_CONVALIBI13_TUNINGL1_VALTEMPORAL_TESTBEST.xls"],
    "LiveFuelMoistureContent": ["output_zerveas/LiveFuelMoistureContent_REPRO_ZERVEAS.xls", "LiveFuelMoistureContent_per_timestep_TUNING_FIXEDSTRIDE_SEQSPLIT_TESTBEST.xls",
                                "LiveFuelMoistureContent_lstm_TUNING_FIXEDSTRIDE_SEQSPLIT_TESTBEST.xls", "LiveFuelMoistureContent_LADAA_TUNING_FIXEDSTRIDE_TESTBEST.xls"],
                                # "LiveFuelMoistureContent_LOCAL_per_timestep_TESTBEST.xls", "LiveFuelMoistureContent_LOCAL_lstm_TESTBEST.xls", "LiveFuelMoistureContent_CONVALIBI14_TUNINGL1_VALTEMPORAL_TESTBEST.xls"],
    "CropYield": ["/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/output/CountyFeatureExtractor_encoder=mvts_transformer_DAILYAUG1_TST/corn_daily_yield/results_summary_FINAL.csv",
                  "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/output/CountyFeatureExtractor_encoder=mvts_local_cnn_DAILYAUG1_PERTIMESTEPMLP/corn_daily_yield/results_summary_FINAL.csv",
                  "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/output/CountyFeatureExtractor_encoder=mvts_local_cnn_DAILYAUG1_LSTM/corn_daily_yield/results_summary_FINAL.csv",
                  "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/output/CountyFeatureExtractor_encoder=mvts_climax_DAILYAUG1_CONVALIBI11_PATCH42_SLOPE025/corn_daily_yield/results_summary_FINAL.csv"]
}
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "analysis_results.csv")
METHODS = ["TST", "Per-Timestep MLP", "LSTM+Attn", "LADAA"]

result_table = []

for dataset, file_list in PAIRED_FILES.items():
    results = []
    for filename in file_list:
        # print("File", os.path.join(OUTPUT_DIR, filename))
        if dataset == "CropYield":  # CropYield file is formatted differently
            df = pd.read_csv(filename)  # .dropna()
            df = df[(df["timestamp"] != "MEAN") & (df["timestamp"] != "STD")]
            test_loss = pd.to_numeric(df['test_rmse_raw'], errors='coerce').dropna().to_numpy()
        else:
            df = pd.read_excel(os.path.join(OUTPUT_DIR, filename), engine='xlrd')  # .dropna()
            df = df[(df["Timestamp"] != "MEAN") & (df["Timestamp"] != "STD")]
            test_loss = pd.to_numeric(df['Test loss'], errors='coerce').dropna().to_numpy()
            test_loss = np.sqrt(test_loss)
        # print("Test loss", test_loss)
        results.append(test_loss)
    
    result_row = []
    for i in range(0, len(METHODS)):
        print(f">>>>>>> {dataset}: TESTING DIFFERENCE BETWEEN {METHODS[0]}, {METHODS[i]}")
        print(f"Values for {METHODS[0]} (Mean {results[0].mean():.3f}, Std {results[0].std(ddof=1):.3f}): {results[0]}")
        print(f"Values for {METHODS[i]} (Mean {results[i].mean():.3f}, Std {results[i].std(ddof=1):.3f}): {results[i]}")
        result = ttest_rel(results[0], results[i])
        statistic = result.statistic
        p_value = result.pvalue
        print("Stat", statistic, f"p={p_value:.3f}")
        rmse_red = 1 - results[i].mean() / results[0].mean()
        print("Reduction RMSE", rmse_red)
        result_row.extend([results[i].mean(), results[i].std(ddof=1), p_value, rmse_red])
    result_table.append(result_row)

row_names = []
for method in METHODS:
    row_names.extend([f"{method}_mean", f"{method}_std", f"{method}_pvalue", f"{method}_rmse_reduction"])
df = pd.DataFrame(np.array(result_table).T, index=row_names, columns=PAIRED_FILES.keys())

# Truncate decimals
df = df.round(decimals=3)
df.to_csv(OUTPUT_FILE)


# rmse_reductions_all = np.array(rmse_reductions_all)
# print("RMSE REDUCTIONS", rmse_reductions_all)
# print("Avg by method", rmse_reductions_all.mean(axis=0))
# pvalues_all = np.array(pvalues_all)
# print("PVALUES", pvalues_all)


# # Sample data representing paired observations (e.g., scores before and after an intervention)
# scores_before = np.array([85, 90, 78, 92, 88])
# scores_after = np.array([91, 94, 88, 96, 87])

# # Perform the Wilcoxon Signed-Rank test
# # The function returns the W-statistic and the p-value
# W_statistic, p_value = wilcoxon(scores_before, scores_after)

# # Print the results
# print(f"Wilcoxon W-statistic: {W_statistic:.2f}")
# print(f"P-value: {p_value:.4f}")

# # Interpretation:
# # If p_value < alpha (e.g., 0.05), reject the null hypothesis,
# # suggesting a significant difference between the paired samples.
# # Otherwise, fail to reject the null hypothesis.
# alpha = 0.05
# if p_value < alpha:
#     print("Reject the null hypothesis: There is a significant difference.")
# else:
#     print("Fail to reject the null hypothesis: No significant difference found.")