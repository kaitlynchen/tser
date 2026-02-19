# Code for paper "Locality And Distance-Aware Attention for Time Series Extrinsic Regression"

The code for Table 1, 3, 4 (TSER archive datasets) are mainly in `mvts_transformer/src`, and the code for Table 2 (crop yield) is in `Crop_Yield_Prediction/baseline`.

## Installation

The TSER experiments were run using Python 3.8, PyTorch 2.3.0, CUDA 12.1 (roughly matching [TST (Zerveas et al. 2021)](https://github.com/gzerveas/mvts_transformer) code). To install the packages we used, 
you can run

```
cd mvts_transformer/src
conda env create -f environment.yml
conda activate tser
```

Alternatively, you can use pip on the TST requirements file (not tested):
```
cd mvts_transformer
pip install -r requirements.txt
```

# 1) Time Series Extrinsic Regression archive datasets

## Download datasets

Download datasets from here: https://zenodo.org/record/3902651#.YB5P0OpOm3s

Place them in a directory, and modify the scripts' DATA_DIR to point to that directory.

## Reproducing Table 1

For all scripts, make sure you modify `DATA_DIR` to point to your data directory, and modify the `ACTIVATE ENVIRONMENT` to match your setup (or delete it).

You can run a script as follows:
`./repro_scripts/SCRIPT.sh <DATASET>`

(replace <DATASET> with one of: AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent)

Alternatively if using slurm, you can submit a job as
`sbatch repro_scripts/SCRIPT.sh <DATASET>`

### LADAA

For hyperparameter tuning, and then testing using the best hyperparameters, run
```
./repro_scripts/ladaa_tuning.sh <DATASET>
```
The hyperparameter tuning runs will be written to `output/<OUTPUT_FILE>.xls` (OUTPUT_FILE is defined in the script).
The final test results using the best hyperparameters will be written to `output/<OUTPUT_FILE>_TESTBEST.xls`.

NOTE that the `Test loss` column is Mean Squared Error (WITHOUT the square root). However, the MEAN and STD values
are after taking the square root.

For just running the best hyperparameters:
```
./repro_scripts/ladaa_final.sh <DATASET>
```


### Per-Timestep MLP and LSTM

For hyperparameter tuning (both per-timestep MLP and LSTM):
```
./repro_scripts/pertimestepmlp_tuning.sh <DATASET>
```

For running the best hyperparameters:
```
./repro_scripts/pertimestepmlp_final.sh <DATASET>
./repro_scripts/lstm_final.sh <DATASET>
```


### ConViT/ConvTran
Hyperparameter tuning and then testing best hyperparameters:
```
./repro_scripts/convit_tuning.sh <DATASET>
./repro_scripts/convtran_tuning.sh <DATASET>
```

## Ablations
For Table 3, 4, 17 run the following:
```
./repro_scripts/ablation/posenc_ablations_Benzene.sh
./repro_scripts/ablation/posenc_ablations_BeijingPM10.sh
./repro_scripts/ablation/posenc_ablations_IEEEPPG.sh
```

For local mask experiment (Table 7):
```
./repro_scripts/repro_test/local_mask.sh <DATASET>
```

For sensitivity analyses (Tables 13-16):
```
./repro_scripts/ablation/sens_patch.sh <DATASET>
./repro_scripts/ablation/sens_l1.sh <DATASET>
./repro_scripts/ablation/sens_slope.sh <DATASET>
./repro_scripts/ablation/sens_smooth.sh <DATASET>
```

For pooling ablation (Table 18):
```
./repro_scripts/ablation/ablation_pool.sh <DATASET>
```

Statistical tests can be run with `python3 statistical_tests.py`, but you would need to change the filenames.

## Code pointers
From the `mvts_transformer/src` directory:
- `main.py` is the entry point.
- `running.py` is the main train loop.
- `models/ts_transformer.py` is the original Zerveas TST model.
- `models/ts_climax.py` contains code for our LADAA and other Transformer variants, including various forms of relative positional encoding, absolute positional encoding, attention smoothness, masks, and pooling.
- `models/local_cnn.py` contains the per-timestep MLP and LSTM models.
- `test_best.py` parses the result of hyperparameter tuning and tests the best config with 3 seeds
Other files are mostly irrelevant.

# 2) Crop Yield experiments

## Download datasets

The data can be downloaded [here](https://osf.io/3qhru/?view_only=f1edbd5af91642cebef79a08c94bf6fd) - you only need to download `combined_dataset_daily_32bit.npz`. Or you can run the following commands:
```
wget https://osf.io/download/6891731c1262b0433ae7cff2/?view_only=f1edbd5af91642cebef79a08c94bf6fd
mv 'index.html?view_only=f1edbd5af91642cebef79a08c94bf6fd' combined_dataset_daily_32bit.npz
unzip 'index.html?view_only=f1edbd5af91642cebef79a08c94bf6fd'
```

Modify the scripts' `DATA_FILE` to point to the `combined_dataset_daily_32bit.npz` file. 

## Reproducing Table 2

These scripts will go through the hyperparameter tuning and testing process. Please ensure `DATA_FILE` and `OUTPUT_DIR` are correct, and
the `ACTIVATE ENVIRONMENT` section is updated. Run from `Crop_Yield_Prediction/baseline` directory.

```
./repro_scripts/ladaa.sh
./repro_scripts/cnn.sh
./repro_scripts/fcn.sh
./repro_scripts/inception.sh
./repro_scripts/tst.sh
./repro_scripts/pertimestepmlp.sh
./repro_scripts/lstm.sh
```

Don't forget to change DATA_FILE, OUTPUT_DIR, and "ACTIVATE ENVIRONMENT" commands. Results will be written to a folder inside OUTPUT_DIR that contains NOTE. `results_summary_fold0_tuning.csv` contains tuning results, and `results_summary_FINAL.csv_withmeanstd.csv` contains final test results across 5 folds. Check the *_raw columns for raw yield metrics (others are deviation from trend).

For an example of testing on a single set of final hyperparameters, try
```
./repro_scripts/ladaa_final.sh
```

## Code pointers
From the `Crop_Yield_Prediction` directory:
- `baseline/single_year_main.py` is the entry point.
- `baseline/single_year_train.py` is the main train script.
- `baseline/test_best.py` parses the result of hyperparameter tuning and tests the best config on all 5 folds.
- `shared_utils/new_models.py` contains most of the wrapper model code.
Other files are mostly irrelevant.