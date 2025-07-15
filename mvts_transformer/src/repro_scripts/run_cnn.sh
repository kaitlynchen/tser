#!/bin/bash

# CNN baseline. Usage:
# ./run_cnn.sh AppliancesEnergy
# # Josh's results:
# 1) CNN_AppliancesEnergy_seqpool_multihead_local_VAL.xls: 3.22 (lr=1e-3), 3.68 (lr=1e-2)
# 2) CNN_AppliancesEnergy_seqpool_multihead_local_TEST.xls: 2.26 (lr=1e-3), 2.13 (lr=1e-2)
# 3) CNN_AppliancesEnergy_seqpool_multihead_per_timestep_VAL.xls: 2.79 (lr=1e-2)
# 4) CNN_AppliancesEnergy_seqpool_multihead_per_timestep_TEST.xls: 2.05 (lr=1e-2)

# Usage on Slurm:
# sbatch repro_scripts/run_cnn.sh
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the full partition
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J tser_cnn
# Request 1 V100 GPU
#SBATCH --gpus v100:1
# Request 4 CPU cores (8 hyperthreads).
#SBATCH -c 8
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 80GB RAM
#SBATCH --mem=40GB
# Request a walltime limit of 72 hours
#SBATCH -t 72:00:00

# Activate environment
source ~/.bashrc
conda activate tser
cd ~/tser/mvts_transformer/src


# Dataset
# IEEEPPG LiveFuelMoistureContent BeijingPM10Quality
DATA="AppliancesEnergy"

# Batch size: 16 for AppliancesEnergy, otherwise 128
if [ "$DATA" = "AppliancesEnergy" ]; then
    BS=16
else
    BS=128
fi
if [ "$DATA" = "IEEEPPG" ]; then
    PATCH=16
    STRIDE=8
else
    PATCH=1
    STRIDE=1
fi

for POOL in seqpool_multihead
do
    for CONV_TYPE in per_timestep
    do
        for WHERE_ABSPOS in before_pool_concat
        do
            for LR in 1e-2 1e-3
            do
                for LAM in 0 1e-3 1e-1 10
                do
                    for SEED in 0 1 2
                    do
                        python main.py --comment "${DATA}_CNN_${POOL}_${CONV_TYPE}_VAL_TUNING" \
                            --seed $SEED --name "${DATA}_CNN_${POOL}_${CONV_TYPE}_VAL_TUNING" \
                            --records_file output/${DATA}_CNN_${POOL}_${CONV_TYPE}_VAL_TUNING.xls \
                            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                            --pattern TRAIN --val_ratio 0.2 --epochs 1000 --patience 200 \
                            --lr $LR --batch_size $BS \
                            --num_heads 16 --d_model 128 \
                            --optimizer RAdam --task regression --normalize_label \
                            --model local_cnn --conv_type $CONV_TYPE --patch_length $PATCH --stride $STRIDE --smooth_attention \
                            --pos_encoding learnable_sin_init --where_to_add_abspos $WHERE_ABSPOS \
                            --pool $POOL --reg_lambda $LAM \
                            --plot_loss --plot_accuracy
                    done
                done
            done
        done
    done
done


# for POOL in seqpool_multihead
# do
#     for CONV_TYPE in per_timestep
#     do
#         for WHERE_ABSPOS in start_add before_pool_concat before_pool_add
#         do
#             for LR in 1e-2 1e-3
#             do
#                 for SEED in 0 1 2
#                 do
#                     python main.py --comment "${DATA}_CNN_${POOL}_${CONV_TYPE}_VAL2_PATCH12_ABSENC" \
#                         --seed $SEED --name "${DATA}_CNN_${POOL}_${CONV_TYPE}_VAL2_PATCH12_ABSENC" \
#                         --records_file output/${DATA}_CNN_${POOL}_${CONV_TYPE}_VAL2_PATCH12_ABSENC.xls \
#                         --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
#                         --pattern TRAIN --val_ratio 0.2 --epochs 500 --patience 100 \
#                         --lr $LR --batch_size $BS \
#                         --num_heads 16 --d_model 128 \
#                         --optimizer RAdam --task regression --normalize_label \
#                         --model local_cnn --conv_type $CONV_TYPE --patch_length 16 --stride 8 \
#                         --pos_encoding learnable_sin_init --where_to_add_abspos $WHERE_ABSPOS \
#                         --pool $POOL \
#                         --plot_loss --plot_accuracy
#                 done
#             done
#         done
#     done
# done










# for POOL in seqpool_multihead
# do
#     for ABSPOS in learnable_sin_init learnable_uniform_init
#     do
#         for ABSPOS_WHERE in before_pool_concat 
#         do
#             for CONV_TYPE in per_timestep
#             do
#                 for LR in 1e-2 1e-3
#                 do
#                     for BS in 16 32
#                     do
#                         for SEED in 0 1 2
#                         do
#                             python main.py --comment "${DATA}_CNN_${POOL}_${CONV_TYPE}_TUNING" \
#                                 --seed $SEED --name "${DATA}_CNN_${POOL}_${CONV_TYPE}_TUNING" \
#                                 --records_file output/${DATA}_CNN_${POOL}_${CONV_TYPE}_TUNING.xls \
#                                 --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
#                                 --pattern TRAIN --val_ratio 0.2 --epochs 500 --patience 100 \
#                                 --lr $LR --batch_size $BS \
#                                 --num_heads 16 --d_model 128 \
#                                 --optimizer RAdam --task regression --normalize_label \
#                                 --model local_cnn --patch_length 1 --stride 1 \
#                                 --pos_encoding $ABSPOS --where_to_add_abspos $ABSPOS_WHERE \
#                                 --pool $POOL --conv_type $CONV_TYPE \
#                                 --plot_loss --plot_accuracy
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# for LR in 1e-4 1e-3 1e-2
# do
#     for POOL in seqpool_multihead
#     do
#         for CONV_TYPE in per_timestep local
#         do
#             for SEED in 0 1 2
#             do
#                 python main.py --comment "${DATA}_CNN_${POOL}_${CONV_TYPE}_VAL" \
#                     --seed $SEED --name "${DATA}_CNN_${POOL}_${CONV_TYPE}_VAL" \
#                     --records_file output/${DATA}_CNN_${POOL}_${CONV_TYPE}_VAL.xls \
#                     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
#                     --pattern TRAIN --val_ratio 0.2 --epochs 500 --patience 100 \
#                     --lr $LR --batch_size $BS \
#                     --num_heads 16 --d_model 256 \
#                     --optimizer RAdam --task regression --normalize_label \
#                     --model local_cnn --conv_type $CONV_TYPE --patch_length 1 --stride 1 \
#                     --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
#                     --pool $POOL \
#                     --plot_loss --plot_accuracy

#                 # python main.py --comment "${DATA}_CNN_${POOL}_${CONV_TYPE}_TEST" \
#                 #     --seed $SEED --name "${DATA}_CNN_${POOL}_${CONV_TYPE}_TEST" \
#                 #     --records_file output/${DATA}_CNN_${POOL}_${CONV_TYPE}_TEST.xls \
#                 #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
#                 #     --pattern TRAIN --val_pattern TEST --epochs 500 --patience 100 \
#                 #     --lr $LR --batch_size $BS \
#                 #     --num_heads 16 --d_model 256 \
#                 #     --optimizer RAdam --task regression --normalize_label \
#                 #     --model local_cnn --conv_type $CONV_TYPE --patch_length 1 --stride 1 \
#                 #     --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
#                 #     --pool $POOL \
#                 #     --plot_loss --plot_accuracy

#                 # python main.py --comment "CNN_${DATA}_${POOL}_${CONV_TYPE}_TEST" \
#                 #     --seed $SEED --name CNN_${DATA} \
#                 #     --records_file output_repro/CNN_20250602_${DATA}_${POOL}_${CONV_TYPE}_TEST.xls \
#                 #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
#                 #     --pattern TRAIN --val_pattern TEST --epochs 500 --lr $LR \
#                 #     --optimizer RAdam --task regression --normalize_label \
#                 #     --model local_cnn --pool $POOL --conv_type $CONV_TYPE --pos_encoding learnable_sin_init  \
#                 #     --plot_loss --plot_accuracy
#             done
#         done
#     done
# done
