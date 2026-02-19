#!/bin/bash

# ============================= USAGE ==============================
# Runs best hyperparameters for per-timestep MLP.

# ./repro_scripts/pertimestepmlp_final.sh <DATASET>
# (replace <DATASET> with one of: AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent)
# Don't forget to change DATA_DIR and "ACTIVATE ENVIRONMENT" commands.

# Results will be written to `output/${OUTPUT_FILE}.xls` (you can modify OUTPUT_FILE in this script).
# NOTE: The "test loss" column is MSE, take square root to get RMSE.

# ======== SLURM BOILERPLATE - Ignore if not using Slurm. =========
# Request the full partition
#SBATCH -p full
#SBATCH --exclude=c0020,c0002
# Name the job so it's meaningful in the job list
#SBATCH -J erpe_convalibi
# Request 1 V100 GPU
#SBATCH --gpus v100:1
# Request 4 CPU cores (8 hyperthreads).
#SBATCH -c 8
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 80GB RAM
#SBATCH --mem=20GB
# Request a walltime limit of 72 hours
#SBATCH -t 72:00:00

# ============================ DATASET ============================
DATA=$1
echo $DATA
DATA_DIR="/mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/"  # TODO: Change this!!!

# ========== ACTIVATE ENVIRONMENT (TODO: Change this!!!) ==========
source ~/.bashrc
conda activate tser

# ======================= HYPERPARAMETERS =========================
# Default settings
BS=128
PATCH=1
STRIDE=1
SPLIT=""

# Dataset-specific overrides
if [ "$DATA" = "AppliancesEnergy" ]; then
    BS=16  # Use smaller batch size for AppliancesEnergy, since there are only 95 train examples
elif [ "$DATA" = "IEEEPPG" ]; then
    PATCH=4
    STRIDE=4
elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
    PATCH=8
    STRIDE=4
fi

# Hyperparams
if [ "$DATA" = "AppliancesEnergy" ]; then
    LR=1e-2
elif [ "$DATA" = "BeijingPM10Quality" ]; then
    LR=1e-4
elif [ "$DATA" = "BeijingPM25Quality" ]; then
    LR=1e-4
elif [ "$DATA" = "BenzeneConcentration" ]; then
    LR=1e-3
elif [ "$DATA" = "IEEEPPG" ]; then
    LR=1e-4
elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
    LR=1e-4
fi

# Test per-timestep MLP
for CONV_TYPE in per_timestep
do
    OUTPUT_FILE="${DATA}_${CONV_TYPE}_FINAL"
    for POOL in seqpool_multihead
    do
        for WHERE_ABSPOS in before_pool_concat
        do
            for SEED in 0 1 2
            do
                PARAM_STR="LR=${LR}_POOLSMOOTH=${LAM}_SEED=${SEED}"
                python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
                    --records_file "output/${OUTPUT_FILE}.xls" \
                    --data_dir $DATA_DIR --data_class tsra \
                    --pattern TRAIN --val_pattern TEST \
                    --epochs 2000 --patience 200 \
                    --lr $LR --batch_size $BS \
                    --num_heads 16 --d_model 128 \
                    --optimizer RAdam --task regression --normalize_label \
                    --model local_cnn --conv_type $CONV_TYPE \
                    --patch_length $PATCH --stride $STRIDE --smooth_attention \
                    --pos_encoding learnable_sin_init --where_to_add_abspos $WHERE_ABSPOS \
                    --pool $POOL \
                    --plot_loss --plot_accuracy
            done
        done
    done
done