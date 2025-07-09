#!/bin/bash

# Usage on Slurm:
# sbatch repro_scripts/erpe_BenzeneConcentration.sh
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the full partition
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J BenzeneTest
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

# Activate environment
source ~/.bashrc
conda activate tser
cd ~/tser/mvts_transformer/src

for DATA in BenzeneConcentration
do
    for LR in 1e-3 1e-2 1e-1
    do
        for HEADS in 8 12 16
        do
            for SEED in 0 1 2
            do
                # BASIC
                python main.py --comment "ClimaX on ${DATA}" \
                    --seed $SEED --name ${DATA}_CONVIT2 \
                    --records_file output_repro/${DATA}_CONVIT2.xls \
                    --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                    --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --lr $LR \
                    --num_layers 3 --num_heads $HEADS --d_model 128 --dim_feedforward 256 \
                    --optimizer RAdam --task regression \
                    --plot_loss --plot_accuracy \
                    --model climax_smooth --patch_length 1 --stride 1 --smooth_attention --normalize_label \
                    --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
                    --relative_pos_encoding convit --where_to_add_relpos only_relpos \
                    --pool seqpool_multihead
            done
        done
    done
done

# for DATA in BenzeneConcentration
# do
#     for LR in 1e-3 1e-2
#     do
#         for LAM in 0 1e-1 10
#         do
#             for SEED in 0 1 2
#             do
#                 # BASIC
#                 python main.py --comment "ClimaX on ${DATA}" \
#                     --seed $SEED --name ${DATA}_ERPECONVIT_ABSPOSCONCAT_TEMP \
#                     --records_file output_repro/${DATA}_ERPECONVIT_ABSPOSCONCAT_TEMP.xls \
#                     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
#                     --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 500 --lr $LR --batch_size 16 \
#                     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
#                     --optimizer RAdam --task regression \
#                     --plot_loss --plot_accuracy \
#                     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention --normalize_label \
#                     --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
#                     --relative_pos_encoding erpe_convit_init --where_to_add_relpos only_relpos \
#                     --pool seqpool_multihead --reg_lambda $LAM
#             done
#         done
#     done
# done

# # Commands to test various absolute position encoding, relative position encoding, pooling, etc. choices
# for DATA in BenzeneConcentration
# do
#     for LR in 1e-3 1e-2
#     do
#         for LAM in 0
#         do
#             for SEED in 0
#             do
#                 # BASIC
#                 python main.py --comment "ClimaX on ${DATA}" \
#                     --seed $SEED --name ${DATA}_CONVIT_ABSPOSCONCAT \
#                     --records_file output_repro/${DATA}_CONVIT_ABSPOSCONCAT.xls \
#                     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
#                     --pattern TRAIN --val_ratio 0.2 --epochs 1000 --patience 200 --lr $LR --batch_size 128 \
#                     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
#                     --optimizer RAdam --task regression \
#                     --plot_loss --plot_accuracy \
#                     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention --normalize_label \
#                     --pos_encoding learnable_uniform_init --where_to_add_abspos before_pool_concat \
#                     --relative_pos_encoding convit --where_to_add_relpos only_relpos \
#                     --pool seqpool_multihead --reg_lambda $LAM
#             done
#         done
#     done
# done
#                     --where_to_add_abspos before_pool_concat \
#                     --relative_pos_encoding erpe_uniform_init --where_to_add_relpos only_relpos \
#                     --pool seqpool_multihead --lambda_posenc_smoothness $LAM

# Commands to test various absolute position encoding, relative position encoding, pooling, etc. choices
for DATA in BenzeneConcentration
do
    for LR in 1e-3 1e-2
    do
        for LAM in 0 10 100
        do
            for SEED in 0
            do
                # TUNING
                python main.py --comment "ClimaX on ${DATA}" \
                    --seed $SEED --name CLIMAX_ONLYRELPOS_CONCATABSPOS_${DATA}_TUNING \
                    --records_file output_repro/CLIMAX_ONLYRELPOS_CONCATABSPOS_${DATA}_TUNING.xls \
                    --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                    --pattern TRAIN --val_ratio 0.2 --epochs 1000 --patience 200 --lr $LR --batch_size 128 \
                    --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                    --optimizer RAdam --task regression \
                    --plot_loss --plot_accuracy \
                    --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
                    --pos_encoding learnable_zero_init --where_to_add_abspos before_pool_concat \
                    --relative_pos_encoding erpe_uniform_init --where_to_add_relpos only_relpos \
                    --pool seqpool_multihead --lambda_posenc_smoothness $LAM
            done
        done
    done
done
