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
#SBATCH --mem=200GB
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
PATIENCE=100

# Dataset-specific overrides
if [ "$DATA" = "AppliancesEnergy" ]; then
    BS=16  # Use smaller batch size for AppliancesEnergy, since there are only 95 train examples
    PATIENCE=500
    SPLIT="--val_sequential_split"
elif [ "$DATA" = "BenzeneConcentration" ]; then
    PATIENCE=200
    SPLIT="--val_sequential_split"
elif [ "$DATA" = "IEEEPPG" ]; then
    PATIENCE=200
    PATCH=16
    STRIDE=8
    SPLIT="--val_sequential_split"
elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
    PATCH=8
    STRIDE=4
    SPLIT="--val_sequential_split"
fi


OUTPUT_DIR="./output/ablation_relpos_init"

for WHERE in only_relpos
do
    for RELPOS in erpe_uniform_init erpe_alibi_init
    do
        OUTPUT_FILE="${DATA}_ABLATION_RELPOS_INIT=${RELPOS}_WHERE=${WHERE}"

        for LR in 1e-3 1e-2
        do
            for CONVIT_SLOPE in 1
            do
                for NOISE in 0
                do
                    for HEADS in 16
                    do
                        for LAM in 0 1e-3
                        do
                            for L1 in 0
                            do
                                for SEED in 0
                                do
                                    PARAM_STR="LR=${LR}_HEADS=${HEADS}_LAM=${LAM}_L1=${L1}"

                                    # Baseline, convalibi init relpos, before softmax
                                    python main.py --output_dir "$OUTPUT_DIR" \
                                        --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}"  \
                                        --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls" \
                                        --data_dir $DATA_DIR --data_class tsra \
                                        --pattern TRAIN --val_ratio 0.2 $SPLIT \
                                        --epochs 2000 --patience $PATIENCE \
                                        --lr $LR --batch_size $BS --input_noise_std $NOISE \
                                        --num_layers 3 --num_heads $HEADS --d_model 128 --dim_feedforward 256 \
                                        --optimizer RAdam --task regression \
                                        --plot_loss --plot_accuracy \
                                        --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
                                        --pos_encoding learnable_sin_init --where_to_add_abspos start_concat \
                                        --relative_pos_encoding $RELPOS --where_to_add_relpos $WHERE --convit_slope $CONVIT_SLOPE --alibi_min_slope 0.1 --alibi_max_slope 100 \
                                        --pool seqpool_multihead --reg_lambda $LAM --reg_lambda_pool $LAM --l1 $L1
                                done
                            done
                        done
                    done
                done
            done
        done

        python test_best.py --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls"

    done
done
