#!/bin/bash

# ============================= USAGE ==============================
# Tunes hyperparameters for LADAA.

# ./repro_scripts/ladaa_tuning.sh <DATASET>
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
PATIENCE=200

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


# OUTPUT FILE
OUTPUT_DIR="./output/LADAA_LOCAL_MASK_CONVITINIT_1LAYER_20260223"

# TUNING
for MASK in 0 2 4
do
    OUTPUT_FILE="${DATA}_LADAA_LOCAL_MASK_CONVITINIT_1LAYER_20260223_MASK=${MASK}"

    for LR in 1e-3 1e-2
    do
        for CONVIT_SLOPE in 0.1 1 10
        do
            for HEADS in 8
            do
                for SMOOTH in 0
                do
                    for L1 in 0
                    do
                        for SEED in 0
                        do
                            # BASIC
                            PARAM_STR="LR=${LR}_POOLSMOOTH=${SMOOTH}_L1=${L1}_SLOPE=${CONVIT_SLOPE}_HEADS=${HEADS}_SEED=${SEED}"
                            python main.py --output_dir "$OUTPUT_DIR" \
                                --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}"  \
                                --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls" \
                                --data_dir $DATA_DIR --data_class tsra \
                                --pattern TRAIN --val_ratio 0.2 $SPLIT \
                                --epochs 2000 --patience $PATIENCE \
                                --lr $LR --batch_size $BS \
                                --num_layers 1 --num_heads $HEADS --d_model 128 --dim_feedforward 256 \
                                --optimizer RAdam --task regression \
                                --plot_loss --plot_accuracy \
                                --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
                                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
                                --relative_pos_encoding erpe_convit_init --where_to_add_relpos only_relpos --convit_slope $CONVIT_SLOPE \
                                --pool seqpool_multihead --reg_lambda_pool $SMOOTH --reg_lambda $SMOOTH --l1_reg $L1 --local_mask $MASK
                        done
                    done
                done
            done
        done
    done

    # TEST
    python test_best.py --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls"

done
# #--alibi_min_slope 0.1 --alibi_max_slope 100 \


# # OUTPUT FILE
# OUTPUT_DIR="./output/LADAAREPRO_20260220_NOMIXUP"
# OUTPUT_FILE="LADAAREPRO_20260220_NOMIXUP_${DATA}"

# # TUNING
# for LR in 1e-3 1e-2
# do
#     for CONVIT_SLOPE in 0.25
#     do
#         for HEADS in 16
#         do
#             for SMOOTH in 0 1e-2 1e-1
#             do
#                 for L1 in 0 1e-4
#                 do
#                     for SEED in 0
#                     do
#                         # BASIC
#                         PARAM_STR="LR=${LR}_POOLSMOOTH=${SMOOTH}_L1=${L1}_SLOPE=${CONVIT_SLOPE}_HEADS=${HEADS}_SEED=${SEED}"
#                         python main.py --output_dir "$OUTPUT_DIR" \
#                             --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}"  \
#                             --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls" \
#                             --data_dir $DATA_DIR --data_class tsra \
#                             --pattern TRAIN --val_ratio 0.2 $SPLIT \
#                             --epochs 2000 --patience $PATIENCE \
#                             --lr $LR --batch_size $BS \
#                             --num_layers 3 --num_heads $HEADS --d_model 128 --dim_feedforward 256 \
#                             --optimizer RAdam --task regression \
#                             --plot_loss --plot_accuracy \
#                             --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
#                             --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
#                             --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos --convit_slope $CONVIT_SLOPE \
#                             --pool seqpool_multihead --reg_lambda_pool $SMOOTH --reg_lambda $SMOOTH --l1_reg $L1
#                     done
#                 done
#             done
#         done
#     done
# done

# # TEST
# python test_best.py --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls"

# # TUNING
# for LR in 1e-3 1e-2
# do
#     for CONVIT_SLOPE in 0.25 0.5 1
#     do
#         for HEADS in 8 16
#         do
#             for NOISE in 0
#             do
#                 for SEED in 0
#                 do
#                     # BASIC
#                     PARAM_STR="LR=${LR}_SLOPE=${CONVIT_SLOPE}_HEADS=${HEADS}_SEED=${SEED}"
#                     python main.py --output_dir "$OUTPUT_DIR" \
#                         --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}"  \
#                         --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls" \
#                         --data_dir $DATA_DIR --data_class tsra \
#                         --pattern TRAIN --val_ratio 0.2 $SPLIT \
#                         --epochs 2000 --patience $PATIENCE \
#                         --lr $LR --batch_size $BS \
#                         --num_layers 3 --num_heads $HEADS --d_model 128 --dim_feedforward 256 \
#                         --optimizer RAdam --task regression \
#                         --plot_loss --plot_accuracy \
#                         --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
#                         --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
#                         --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos --convit_slope $CONVIT_SLOPE \
#                         --pool seqpool_cls --input_noise_std $NOISE
#                 done
#             done
#         done
#     done
# done

# # TEST
# python test_best.py --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls"



# # TUNING
# for LR in 1e-3 1e-2
# do
#     for CONVIT_SLOPE in 0.25
#     do
#         for HEADS in 16
#         do
#             for SMOOTH in 1e-2 1e-1
#             do
#                 for L1 in 0 1e-4 1e-3
#                 do
#                     for SEED in 0
#                     do
#                         # BASIC
#                         PARAM_STR="LR=${LR}_POOLSMOOTH=${LAM}_L1=${LAM2}_SLOPE=${CONVIT_SLOPE}_HEADS=${HEADS}_SEED=${SEED}"
#                         python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
#                             --records_file "output/${OUTPUT_FILE}.xls" \
#                             --data_dir $DATA_DIR --data_class tsra \
#                             --pattern TRAIN --val_ratio 0.2 $SPLIT \
#                             --epochs 2000 --patience $PATIENCE \
#                             --lr $LR --batch_size $BS \
#                             --num_layers 3 --num_heads $HEADS --d_model 128 --dim_feedforward 256 \
#                             --optimizer RAdam --task regression \
#                             --plot_loss --plot_accuracy \
#                             --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
#                             --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
#                             --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos --convit_slope $CONVIT_SLOPE \
#                             --pool seqpool_multihead --reg_lambda_pool $SMOOTH --reg_lambda $SMOOTH --l1_reg $L1
#                     done
#                 done
#             done
#         done
#     done
# done

