#!/bin/bash

# ============================= USAGE ==============================
# Scikit-learn baselines. Usage:
# ./run_sklearn.sh <DATASET>
# OR
# sbatch run_sklearn.sh <DATASET>
# (replace <DATASET> with one of: AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent)
# Don't forget to change DATA_DIR and "ACTIVATE ENVIRONMENT" commands.

# Results will be written to `${OUTPUT_DIR}/${OUTPUT_FILE}.xls` (you can modify them in this script).
# NOTE: The "test loss" column is MSE, take square root to get RMSE

# ======== SLURM BOILERPLATE - Ignore if not using Slurm. =========
# Request the full partition
#SBATCH -p full
#SBATCH --exclude=c0020,c0002
# Name the job so it's meaningful in the job list
#SBATCH -J tser_sklearn
# Request 2 CPU cores (4 hyperthreads).
#SBATCH -c 4
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 80GB RAM
#SBATCH --mem=80GB
# Request a walltime limit of 72 hours
#SBATCH -t 72:00:00

# ============================ DATASET ============================
DATA=$1
echo $DATA
DATA_DIR="/mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/"  # TODO: Change this!!!

# ========== ACTIVATE ENVIRONMENT (TODO: Change this!!!) ==========
source ~/.bashrc
conda activate tser


OUTPUT_DIR="./output/SIMPLE_AVGPOOL"
OUTPUT_FILE="${DATA}_SIMPLE_AVGPOOL"

# for MODEL in lasso ridge xgboost random_forest
for MODEL in xgboost
do
    for SEED in 0 1 2
    do
        python main_sklearn.py --output_dir "$OUTPUT_DIR" \
            --seed $SEED --name "${OUTPUT_FILE}" \
            --records_file "$OUTPUT_DIR/${OUTPUT_FILE}.xls" \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --pattern TRAIN --test_pattern TEST --val_ratio 0  \
            --task regression --model $MODEL --input_pooling_patch 6 --input_pooling_stride 6

    done
done