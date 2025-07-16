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


# Best hyperparams for AppliancesEnergy
# Note 2000 epochs are needed.
# Run `mkdir output_repro` first.
for DATA in AppliancesEnergy
do
    for SEED in 0 
    do
        # # Original Zerveas TST: choosing model by test loss
        # python main.py --comment "TST BASELINE on ${DATA}" \
        #     --seed $SEED --name REPROTEST_MVTS_${DATA}_TEST \
        #     --records_file output_repro/REPROTEST_MVTS_${DATA}_TEST.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_pattern TEST --epochs 2000 --lr 0.001 --batch_size 128 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
        #     --optimizer RAdam --pos_encoding learnable --task regression \
        #     --model transformer --plot_loss --plot_accuracy

        # Climax-Smooth: chooosing model by test loss. Should give same result as second command.
        python main.py --comment "ClimaX on ${DATA}" \
            --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST \
            --records_file output_repro/REPROTEST_CLIMAX_${DATA}_TEST.xls \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --pattern TRAIN --val_pattern TEST --epochs 2000 --lr 1e-2 --batch_size 16 \
            --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
            --optimizer RAdam --pos_encoding learnable --task regression \
            --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
            --plot_loss --plot_accuracy
            # --relative_pos_encoding erpe --where_to_add_relpos after

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
