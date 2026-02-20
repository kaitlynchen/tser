#!/bin/bash

# ============================= USAGE ==============================
# Best attempt to reproduce Zerveas TST results.

# ./repro_scripts/repro_tst/repro_tst.sh <DATASET>
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
if [ "$DATA" = "AppliancesEnergy" ]; then
    D_MODEL=128
    D_FEEDFORWARD=512
    PATIENCE=2000
elif [ "$DATA" = "BenzeneConcentration" ]; then
    D_MODEL=128
    D_FEEDFORWARD=256
    PATIENCE=2000
elif [ "$DATA" = "BeijingPM10Quality" ]; then
    D_MODEL=64
    D_FEEDFORWARD=256
    PATIENCE=500
elif [ "$DATA" = "BeijingPM25Quality" ]; then
    D_MODEL=128
    D_FEEDFORWARD=256
    PATIENCE=500
elif [ "$DATA" = "LiveFuelMoistureContent" ] || [ "$DATA" = "LiveFuel2" ]; then
    D_MODEL=64
    D_FEEDFORWARD=256
    PATIENCE=500
elif [ "$DATA" = "IEEEPPG" ]; then
    D_MODEL=512
    D_FEEDFORWARD=512
    PATIENCE=500
else
    echo "Invalid value of DATA $DATA !"
    continue
fi

if [ "$DATA" = "IEEEPPG" ]; then
    BS=32
else
    BS=128
fi


# OUTPUT_FILE="${DATA}_REPRO_ZERVEAS2000"

# for SEED in 0 1 2
# do
#     PARAM_STR="SEED=${SEED}"
#     python main.py --output_dir "./output/output_zerveas2" \
#         --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
#         --records_file "output/output_zerveas2/${OUTPUT_FILE}.xls" \
#         --data_dir $DATA_DIR --data_class tsra \
#         --pattern TRAIN --val_pattern TEST \
#         --epochs 2000 --patience $PATIENCE \
#         --lr 0.001 --batch_size $BS \
#         --num_layers 3 --num_heads 8 --d_model $D_MODEL --dim_feedforward $D_FEEDFORWARD \
#         --optimizer RAdam --task regression \
#         --model transformer --pos_encoding learnable \
#         --plot_loss --plot_accuracy
# done

OUTPUT_DIR="./output/output_zerveas2"
OUTPUT_FILE="${DATA}_REPRO_ZERVEAS2000_CLIMAX_plotevery200"

for SEED in 0 1 2
do
    # Replication of Zerveas model using our "Climax" codebase
    PARAM_STR="SEED=${SEED}"
    python main.py --output_dir "$OUTPUT_DIR" \
        --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}"  \
        --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls" \
        --data_dir $DATA_DIR --data_class tsra \
        --pattern TRAIN --val_pattern TEST \
        --epochs 2000 --patience $PATIENCE \
        --lr 0.001 --batch_size $BS \
        --num_layers 3 --num_heads 8 --d_model $D_MODEL --dim_feedforward $D_FEEDFORWARD \
        --optimizer RAdam --task regression \
        --model climax_smooth --pos_encoding learnable --patch_length 1 --stride 1 --smooth_attention \
        --plot_loss --plot_accuracy 
done