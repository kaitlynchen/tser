#!/bin/bash
# ^This line must be included above.
# To submit this as a Slurm batch job, run "sbatch repro_scripts/repro_BenzeneConcentration.sh"
# You could also run this from an interactive job, as "./repro_scripts/repro_BenzeneConcentration.sh"


# ================================== SLURM SETTINGS ======================================
# On aida, the full partition contains GPU nodes, while the regular partition only contains CPU.
# For now CPU seems faster than GPU so use CPU.
#SBATCH -p full
#SBATCH --exclude=c0020,c0002

# Name the job so it's meaningful in the job list
#SBATCH -J mvts_benzene
# Request 1 V100 GPU
#SBATCH --gpus v100:1
# Request 2 CPU cores (4 hyperthreads).
#SBATCH -c 4
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 20GB RAM
#SBATCH --mem=20GB
# Request a walltime limit of 24 hours
#SBATCH -t 24:00:00


# ================================= SETUP ENVIRONMENT ======================================
# Substitute what you need to set up your environment
cd ~/tser/mvts_transformer/src
source ~/.bashrc
conda activate tser


# ================================= TRAINING COMMANDS =====================================
# Best hyperparams for BenzeneConcentration
# Run `mkdir output_repro` first.
for DATA in BenzeneConcentration
do
    for SEED in 0 1 2
    do
        # Original Zerveas TST: choosing model by test loss
        python main.py --comment "TST BASELINE on ${DATA}" \
            --seed $SEED --name REPROTEST_MVTS_${DATA}_TEST \
            --records_file output_repro/REPROTEST_MVTS_${DATA}_TEST.xls \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 2000 --lr 0.001 --batch_size 128 \
            --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
            --optimizer RAdam --pos_encoding learnable --task regression \
            --model transformer --plot_loss --plot_accuracy

        # # Original Zerveas TST: choosing model by val loss
        # python main.py --comment "TST BASELINE on ${DATA}" \
        #     --seed $SEED --name REPROTEST_MVTS_${DATA}_VAL \
        #     --records_file output_repro/REPROTEST_MVTS_${DATA}_VAL.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_ratio 0.2 --epochs 1500 --lr 0.001 --batch_size 128 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
        #     --optimizer RAdam --pos_encoding learnable --task regression \
        #     --model transformer --plot_loss --plot_accuracy

        # # Climax-Smooth: chooosing model by test loss. Should give same result as first command.
        # python main.py --comment "ClimaX on ${DATA}" \
        #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST \
        #     --records_file REPROTEST_CLIMAX_${DATA}_TEST.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_pattern TEST --epochs 2000 --lr 0.001 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
        #     --optimizer RAdam --pos_encoding learnable --task regression \
        #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
        #     --plot_loss --plot_accuracy

        # # Climax-Smooth: chooosing model by val loss. Should give same result as second command.
        # python main.py --comment "ClimaX on ${DATA}" \
        #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_VAL \
        #     --records_file REPROTEST_CLIMAX_${DATA}_VAL.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr 0.001 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
        #     --optimizer RAdam --pos_encoding learnable --task regression \
        #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
        #     --plot_loss --plot_accuracy



        # # Try SeqPool
        # python main.py --comment "ClimaX on ${DATA} SEQPOOL" \
        #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST_SEQPOOL_LOCALMASK \
        #     --records_file REPROTEST_CLIMAX_${DATA}_TEST_SEQPOOL_LOCALMASK.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_pattern TEST --epochs 500 --lr 1e-2 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
        #     --optimizer RAdam --pos_encoding learnable_sin_init --task regression \
        #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
        #     --plot_loss --plot_accuracy --normalize_label --pool seqpool_multihead --local_mask 3
    done
done
