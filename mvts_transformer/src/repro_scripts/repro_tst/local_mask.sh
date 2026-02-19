#!/bin/bash

# ============================= USAGE ==============================
# Local mask experiment.

# ./repro_scripts/repro_tst/local_mask.sh <DATASET>
# (replace <DATASET> with one of: AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent)
# Don't forget to change DATA_DIR and "ACTIVATE ENVIRONMENT" commands.

# Results will be written to `output/${OUTPUT_FILE}.xls` (you can modify OUTPUT_FILE in this script).
# NOTE: The "test loss" column is MSE, take square root to get RMSE

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
# Dataset
if [ "$DATA" = "AppliancesEnergy" ]; then
    D_MODEL=128
    D_FEEDFORWARD=512
    PATIENCE=500
elif [ "$DATA" = "BenzeneConcentration" ]; then
    D_MODEL=128
    D_FEEDFORWARD=256
    PATIENCE=500
elif [ "$DATA" = "BeijingPM10Quality" ]; then
    D_MODEL=64
    D_FEEDFORWARD=256
    PATIENCE=100
elif [ "$DATA" = "BeijingPM25Quality" ]; then
    D_MODEL=64
    D_FEEDFORWARD=256
    PATIENCE=100
elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
    D_MODEL=64
    D_FEEDFORWARD=256
    PATIENCE=100
elif [ "$DATA" = "IEEEPPG" ]; then
    D_MODEL=512
    D_FEEDFORWARD=512
    PATIENCE=100
else
    echo "Invalid value of DATA $DATA !"
    continue
fi

for MASK in -1 0 2 4 8
do
    OUTPUT_FILE="${DATA}_LOCALMASK=${MASK}"

    for SEED in 0 1 2
    do
        PARAM_STR="SEED=${SEED}"
        python main.py --output_dir "./output/output_zerveas" \
            --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
            --records_file "output/output_zerveas/${OUTPUT_FILE}.xls" \
            --data_dir $DATA_DIR --data_class tsra \
            --pattern TRAIN --val_pattern TEST \
            --epochs 2000 --patience $PATIENCE \
            --lr 0.001 --batch_size 128 \
            --num_layers 3 --num_heads 8 --d_model $D_MODEL --dim_feedforward $D_FEEDFORWARD \
            --optimizer RAdam --task regression \
            --model climax_smooth --patch_length 1 --stride 1 --smooth_attention\
            --pos_encoding learnable_uniform_init --where_to_add_abspos start_add \
            --pool seqpool_multihead --local_mask $MASK \
            --plot_loss --plot_accuracy
    done
done
