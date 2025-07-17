#!/bin/bash

# Usage on Slurm:
# sbatch repro_scripts/run_erpe_convalibi_init.sh
# Output will appear in a file 'slurm-N.out' where N is the job ID.
# Change dataset in DATA below.

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

# Activate environment
source ~/.bashrc
conda activate tser
cd ~/tser/mvts_transformer/src


# Dataset
# AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent 
# DATA="LiveFuelMoistureContent"
DATA="BenzeneConcentration"  
# Batch size: 16 for AppliancesEnergy, otherwise 128
if [ "$DATA" = "AppliancesEnergy" ]; then
    BS=16
else
    BS=128
fi
if [ "$DATA" = "IEEEPPG" ] || ["$DATA" = "LiveFuelMoistureContent"]; then
    PATCH=16
    STRIDE=8
else
    PATCH=1
    STRIDE=1
fi


# ERPE Convalibi Init

for LR in 1e-2 1e-3
do
    for WD in 0 1e-5 1e-3
    do
        for LAM in 0
        do
            for SEED in 0 1 2
            do
                # BASIC
                python main.py --comment "${DATA}_CONVALIBI3_TUNING_LR=${LR}_SMOOTH=${LAM}_WD=${WD}" \
                    --seed $SEED --name "${DATA}_CONVALIBI3_TUNING_LR=${LR}_SMOOTH=${LAM}_WD=${WD}" \
                    --records_file "output/${DATA}_CONVALIBI3_TUNING_WD.xls" \
                    --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                    --pattern TRAIN --val_ratio 0.2 --epochs 2000 --patience 500 \
                    --lr $LR --batch_size $BS \
                    --global_reg --l2_reg $WD \
                    --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                    --optimizer RAdam --task regression \
                    --plot_loss --plot_accuracy \
                    --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
                    --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
                    --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                    --pool seqpool_multihead --reg_lambda $LAM
            done
        done
    done
done


# # ERPE Convalibi Init
# for MASK in 1 8 16
# do
#     for LR in 1e-2 1e-3
#     do
#         for LAM in 0
#         do
#             for SEED in 0 1 2
#             do
#                 # BASIC
#                 python main.py --comment "${DATA}_CONVALIBIMASK_TUNING_LR=${LR}_SMOOTH=${LAM}" \
#                     --seed $SEED --name "${DATA}_CONVALIBIMASK_TUNING_LR=${LR}_SMOOTH=${LAM}" \
#                     --records_file "output/${DATA}_CONVALIBI3_TUNING.xls" \
#                     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
#                     --pattern TRAIN --val_ratio 0.2 --epochs 1000 --patience 200 \
#                     --lr $LR --batch_size $BS \
#                     --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
#                     --optimizer RAdam --task regression \
#                     --plot_loss --plot_accuracy \
#                     --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
#                     --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
#                     --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
#                     --pool seqpool_multihead --reg_lambda $LAM --local_mask $MASK
#             done
#         done
#     done
# done
