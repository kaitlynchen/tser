#!/bin/bash

# ============================= USAGE ==============================
# Tunes hyperparameters for Convit.

# ./repro_scripts/convit.sh <DATASET>
# (replace <DATASET> with one of: AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent)
# Don't forget to change DATA_DIR and "ACTIVATE ENVIRONMENT" commands.

# Results will be written to `output/${OUTPUT_FILE}.xls` (you can modify OUTPUT_FILE in this script).
# Then, we test the best hyperparameters, and write result to `output/${OUTPUT_FILE}_TESTBEST.xls`
# NOTE: The "test loss" column is MSE, take square root to get RMSE (except for MEAN/STD rows, which are RMSE)

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
DATA_DIR="/mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/"  # TODO: Change this!!!
DATA=$1
echo $DATA

# ========== ACTIVATE ENVIRONMENT (TODO: Change this!!!) ==========
source ~/.bashrc
conda activate tser

# ======================= HYPERPARAMETERS =========================
# Default settings
SPLIT=""

# Batch size: 16 for AppliancesEnergy, otherwise 128
if [ "$DATA" = "AppliancesEnergy" ]; then
    BS=16
elif [ "$DATA" = "IEEEPPG" ]; then
    BS=32
else
    BS=128
fi
if [ "$DATA" = "IEEEPPG" ]; then
    PATCH=8
    STRIDE=4
elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
    PATCH=4
    STRIDE=4
else
    PATCH=1
    STRIDE=1
fi
if [ "$DATA" = "BenzeneConcentration" ]; then
    D_MODEL=128
    DIM_FEEDFORWARD=256
elif [ "$DATA" = "AppliancesEnergy" ]; then
    D_MODEL=128
    DIM_FEEDFORWARD=512
elif [ "$DATA" = "IEEEPPG" ]; then
    D_MODEL=512
    DIM_FEEDFORWARD=512
else
    D_MODEL=64
    DIM_FEEDFORWARD=256
fi

# OUTPUT FILE
OUTPUT_FILE="${DATA}_CONVIT_TUNING"

# TUNING
for LR in 1e-2 1e-3 1e-4
do
    for CONVIT_SLOPE in 0.25
    do
        for HEADS in 16 8
        do
            for SEED in 0
            do
                # BASIC
                PARAM_STR="LR=${LR}_SLOPE=${CONVIT_SLOPE}_HEADS=${HEADS}_SEED=${SEED}"
                python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
                    --records_file "output/${OUTPUT_FILE}.xls" \
                    --data_dir $DATA_DIR --data_class tsra \
                    --pattern TRAIN --val_ratio 0.2 $SPLIT \
                    --epochs 2000 --patience 200 \
                    --lr $LR --batch_size $BS \
                    --num_layers 3 --num_heads $HEADS --d_model $D_MODEL --dim_feedforward $DIM_FEEDFORWARD \
                    --optimizer RAdam --task regression \
                    --plot_loss --plot_accuracy \
                    --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
                    --pos_encoding learnable_sin_init --where_to_add_abspos start_add \
                    --relative_pos_encoding convit --where_to_add_relpos after_gating --convit_slope $CONVIT_SLOPE \
                    --pool seqpool_multihead
            done
        done
    done
done

# TEST
python test_best.py --records_file "output/${OUTPUT_FILE}.xls"
