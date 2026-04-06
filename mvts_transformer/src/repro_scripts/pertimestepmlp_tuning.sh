#!/bin/bash

# ============================= USAGE ==============================
# Tunes hyperparameters for per-timestep MLP and LSTM.

# ./repro_scripts/pertimestepmlp_tuning.sh <DATASET>
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

# Dataset-specific overrides
if [ "$DATA" = "AppliancesEnergy" ]; then
    BS=16  # Use smaller batch size for AppliancesEnergy, since there are only 95 train examples
    SPLIT="--val_sequential_split"
elif [ "$DATA" = "BenzeneConcentration" ]; then
    SPLIT="--val_sequential_split"
elif [ "$DATA" = "IEEEPPG" ]; then
    PATCH=16
    STRIDE=8
    SPLIT="--val_sequential_split"
elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
    PATCH=8
    STRIDE=4
    SPLIT="--val_sequential_split"
fi


# Test abspos variants, seqpool_cls
for CONV_TYPE in per_timestep
do
    for POOL in seqpool_multihead
    do
        for WHERE_ABSPOS in start_concat
        do
            OUTPUT_DIR="./output/TIMESTEP_20260220"
            OUTPUT_FILE="TIMESTEP_20260220_NOMIXUP_${DATA}_${CONV_TYPE}_POOL=${POOL}_ABSPOS=${WHERE_ABSPOS}"

            for LR in 1e-4 1e-3 1e-2
            do
                for NOISE in 0
                do
                    for SEED in 0
                    do
                        PARAM_STR="LR=${LR}_SEED=${SEED}"
                        python main.py --output_dir "$OUTPUT_DIR" \
                            --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}"  \
                            --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls" \
                            --data_dir $DATA_DIR --data_class tsra \
                            --pattern TRAIN --val_ratio 0.2 $SPLIT \
                            --epochs 2000 --patience 200 \
                            --lr $LR --batch_size $BS --input_noise_std $NOISE \
                            --num_heads 16 --d_model 256 \
                            --optimizer RAdam --task regression --normalize_label \
                            --model local_cnn --conv_type $CONV_TYPE \
                            --patch_length $PATCH --stride $STRIDE --smooth_attention \
                            --pos_encoding learnable_sin_init --where_to_add_abspos $WHERE_ABSPOS \
                            --pool $POOL \
                            --plot_loss --plot_accuracy
                    done
                done
            done
            python test_best.py --records_file "${OUTPUT_DIR}/${OUTPUT_FILE}.xls"

        done
    done
done



# # Jacobian loss
# for CONV_TYPE in per_timestep
# do
#     OUTPUT_FILE="${DATA}_${CONV_TYPE}_JACOBIAN"
#     for POOL in seqpool_cls
#     do
#         for WHERE_ABSPOS in start_concat
#         do
#             for BS in 16
#             do
#                 for LR in 1e-4 1e-3 1e-2
#                 do
#                     for LAM in 1 1e-2 1e-4 0
#                     do
#                         for SEED in 0
#                         do
#                             PARAM_STR="BS=${BS}_LR=${LR}_LAM=${LAM}_SEED=${SEED}"
#                             python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
#                                 --records_file "output/${OUTPUT_FILE}.xls" \
#                                 --data_dir $DATA_DIR --data_class tsra \
#                                 --pattern TRAIN --val_ratio 0.2 $SPLIT \
#                                 --epochs 2000 --patience 200 \
#                                 --lr $LR --batch_size $BS --input_noise_std 0.1 --lambda_jacobian $LAM \
#                                 --num_heads 8 --d_model 256 \
#                                 --optimizer RAdam --task regression --normalize_label \
#                                 --model local_cnn --conv_type $CONV_TYPE \
#                                 --patch_length $PATCH --stride $STRIDE --smooth_attention \
#                                 --pos_encoding learnable_sin_init --where_to_add_abspos $WHERE_ABSPOS \
#                                 --pool $POOL \
#                                 --plot_loss --plot_accuracy
#                         done
#                     done
#                 done
#             done
#         done
#     done

#     python test_best.py --records_file "output/${OUTPUT_FILE}.xls"
# done





# # Run both per-timestep MLP and LSTM
# for CONV_TYPE in per_timestep
# do
#     OUTPUT_FILE="${DATA}_${CONV_TYPE}_PATCH_TUNING"
#     for POOL in seqpool_cls
#     do
#         for WHERE_ABSPOS in before_pool_concat
#         do
#             for PATCH in 1 4 8 16 32
#             do
#                 for LR in 1e-4 1e-3 1e-2
#                 do
#                     for SEED in 0
#                     do
#                         PARAM_STR="LR=${LR}_POOLSMOOTH=${LAM}_SEED=${SEED}"
#                         python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
#                             --records_file "output/${OUTPUT_FILE}.xls" \
#                             --data_dir $DATA_DIR --data_class tsra \
#                             --pattern TRAIN --val_ratio 0.2 $SPLIT \
#                             --epochs 2000 --patience 200 \
#                             --lr $LR --batch_size $BS \
#                             --num_heads 16 --d_model 128 \
#                             --optimizer RAdam --task regression --normalize_label \
#                             --model local_cnn --conv_type $CONV_TYPE \
#                             --patch_length $PATCH --stride $PATCH --smooth_attention \
#                             --pos_encoding learnable_sin_init --where_to_add_abspos $WHERE_ABSPOS \
#                             --pool $POOL \
#                             --plot_loss --plot_accuracy
#                     done
#                 done
#             done

#             for LR in 1e-4 1e-3 1e-2
#             do
#                 for SEED in 0
#                 do
#                     PARAM_STR="LR=${LR}_POOLSMOOTH=${LAM}_SEED=${SEED}"
#                     python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
#                         --records_file "output/${OUTPUT_FILE}.xls" \
#                         --data_dir $DATA_DIR --data_class tsra \
#                         --pattern TRAIN --val_ratio 0.2 $SPLIT \
#                         --epochs 2000 --patience 200 \
#                         --lr $LR --batch_size $BS \
#                         --num_heads 16 --d_model 128 \
#                         --optimizer RAdam --task regression --normalize_label \
#                         --model local_cnn --conv_type $CONV_TYPE \
#                         --patch_length 16 --stride 8 --smooth_attention \
#                         --pos_encoding learnable_sin_init --where_to_add_abspos $WHERE_ABSPOS \
#                         --pool $POOL \
#                         --plot_loss --plot_accuracy
#                 done
#             done

#             for LR in 1e-4 1e-3 1e-2
#             do
#                 for SEED in 0
#                 do
#                     PARAM_STR="LR=${LR}_POOLSMOOTH=${LAM}_SEED=${SEED}"
#                     python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
#                         --records_file "output/${OUTPUT_FILE}.xls" \
#                         --data_dir $DATA_DIR --data_class tsra \
#                         --pattern TRAIN --val_ratio 0.2 $SPLIT \
#                         --epochs 2000 --patience 200 \
#                         --lr $LR --batch_size $BS \
#                         --num_heads 16 --d_model 128 \
#                         --optimizer RAdam --task regression --normalize_label \
#                         --model local_cnn --conv_type $CONV_TYPE \
#                         --patch_length 32 --stride 16 --smooth_attention \
#                         --pos_encoding learnable_sin_init --where_to_add_abspos $WHERE_ABSPOS \
#                         --pool $POOL \
#                         --plot_loss --plot_accuracy
#                 done
#             done
#         done
#     done

#     python test_best.py --records_file "output/${OUTPUT_FILE}.xls"
# done