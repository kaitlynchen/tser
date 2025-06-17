#!/bin/bash

# Usage on Slurm:
# sbatch repro_scripts/erpe_AppliancesEnergy.sh
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the full partition
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J erpe_test
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

# Commands to test various absolute position encoding, relative position encoding, pooling, etc. choices
for DATA in AppliancesEnergy
do
    for LR in 1e-3 1e-2
    do
        for LAM in 0 1e-3 1e-2 1e-1 1
        do
            for SEED in 0 1 2
            do
                # Base Climax-Smooth model.
                # To modify the absolute position encoding, you can change --pos_encoding to
                # learnable_zero_init, learnable_sin_init, 
                # python main.py --comment "ClimaX on ${DATA}" \
                #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST \
                #     --records_file output_repro/REPROTEST_CLIMAX_${DATA}_TEST.xls \
                #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                #     --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
                #     --optimizer RAdam --pos_encoding learnable_uniform_init --task regression \
                #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
                #     --plot_loss --plot_accuracy

                # Variable Aggregation + Patching + ERPE + Pooling
                python main.py --comment "ClimaX on ${DATA}" \
                    --seed $SEED --name CLIMAX_ONLYRELPOS_POOLABSPOS_${DATA}_TUNING \
                    --records_file output_repro/CLIMAX_ONLYRELPOS_POOLABSPOS_${DATA}_TUNING.xls \
                    --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                    --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                    --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
                    --optimizer RAdam --task regression \
                    --plot_loss --plot_accuracy \
                    --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
                    --pos_encoding learnable_uniform_init --where_to_add_abspos pooling_before_softmax \
                    --relative_pos_encoding erpe_convit_init --where_to_add_relpos only_relpos \
                    --pool seqpool_multihead --lambda_posenc_smoothness $LAM

                # --agg_vars --patch_length 8 --stride 4 \

                # conv_transformer
                # conv_projection


                # # Same as above but with --normalize_label
                # python main.py --comment "ClimaX on ${DATA}" \
                #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST_NORMALIZELABEL \
                #     --records_file REPROTEST_CLIMAX_${DATA}_TEST.xls \
                #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                #     --pattern TRAIN --val_pattern TEST --epochs 2000 --lr 0.001 --batch_size 128 \
                #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
                #     --optimizer RAdam --pos_encoding learnable --task regression \
                #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
                #     --plot_loss --plot_accuracy --normalize_label

                # # Original Zerveas TST: choosing model by val loss
                # python main.py --comment "TST BASELINE on ${DATA}" \
                #     --seed $SEED --name REPROTEST_MVTS_${DATA}_VAL \
                #     --records_file output_repro/REPROTEST_MVTS_${DATA}_VAL.xls \
                #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                #     --pattern TRAIN --val_ratio 0.2 --epochs 1500 --lr 0.001 --batch_size 128 \
                #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
                #     --optimizer RAdam --pos_encoding learnable --task regression \
                #     --model transformer --plot_loss --plot_accuracy

                # # Climax-Smooth: chooosing model by val loss. Should give same result as first command.
                # python main.py --comment "ClimaX on ${DATA}" \
                #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_VAL \
                #     --records_file REPROTEST_CLIMAX_${DATA}_VAL.xls \
                #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                #     --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr 0.001 \
                #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                #     --optimizer RAdam --pos_encoding learnable --task regression \
                #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
                #     --plot_loss --plot_accuracy
            done
        done
    done
done
