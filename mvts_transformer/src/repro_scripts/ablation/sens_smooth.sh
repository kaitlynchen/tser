#!/bin/bash

# ============================= USAGE ==============================
# Test how changing attention smoothnes loss weight impacts results.

# ./repro_scripts/ablation/sens_smooth.sh <DATASET>
# (replace <DATASET> with one of: AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent)
# Don't forget to change DATA_DIR and "ACTIVATE ENVIRONMENT" commands.

# Results will be written to `output/${OUTPUT_FILE}.xls` (you can modify OUTPUT_FILE in this script).
# NOTE: The "test loss" column is MSE, take square root to get RMSE.

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
# Activate environment
source ~/.bashrc
conda activate tser

# Normal arguments
BS=128
HEADS=16
PATCH=1
STRIDE=1
PATIENCE=100

# Dataset-specific overrides
if [ "$DATA" = "AppliancesEnergy" ]; then
    BS=16  # Use smaller batch size for AppliancesEnergy, since there are only 95 train examples
    PATIENCE=500
elif [ "$DATA" = "BenzeneConcentration" ]; then
    PATIENCE=500
elif [ "$DATA" = "IEEEPPG" ]; then
    PATIENCE=200
    PATCH=8
    STRIDE=4
elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
    PATCH=4
    STRIDE=4
fi

# DEFAULT Hyperparams
if [ "$DATA" = "AppliancesEnergy" ]; then
    LR=1e-2
    CONVIT_SLOPE=0.25
    SMOOTH=1e-3
    L1=1e-3
elif [ "$DATA" = "BeijingPM10Quality" ]; then
    LR=1e-3
    CONVIT_SLOPE=0.25
    SMOOTH=1e-2
    L1=1e-2
elif [ "$DATA" = "BeijingPM10Quality" ]; then
    LR=1e-3
    CONVIT_SLOPE=0.25
    SMOOTH=1e-2
    L1=0
elif [ "$DATA" = "BenzeneConcentration" ]; then
    LR=1e-3
    CONVIT_SLOPE=0.25
    SMOOTH=1e-1
    L1=0
elif [ "$DATA" = "IEEEPPG" ]; then
    LR=1e-2
    CONVIT_SLOPE=0.25
    SMOOTH=1e-1
    L1=1e-2
elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
    LR=1e-2
    CONVIT_SLOPE=0.25
    SMOOTH=1e-1
    L1=1e-3
fi



# OUTPUT FILE
OUTPUT_FILE="${DATA}_SENS_SMOOTH"

for SMOOTH in 0 1e-4 1e-3 1e-2 1e-1
do
    for SEED in 0 1 2
    do
        # BASIC
        PARAM_STR="LR=${LR}_LSMOOTH=${SMOOTH}_L1=${L1}_SLOPE=${CONVIT_SLOPE}_HEADS=${HEADS}_SEED=${SEED}"
        python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
            --records_file "output/${OUTPUT_FILE}.xls" \
            --data_dir $DATA_DIR --data_class tsra \
            --pattern TRAIN --val_pattern TEST \
            --epochs 2000 --patience $PATIENCE \
            --lr $LR --batch_size $BS \
            --num_layers 3 --num_heads $HEADS --d_model 128 --dim_feedforward 256 \
            --optimizer RAdam --task regression \
            --plot_loss --plot_accuracy \
            --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
            --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
            --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos --convit_slope $CONVIT_SLOPE \
            --pool seqpool_multihead --reg_lambda_pool $SMOOTH --reg_lambda $SMOOTH --l1_reg $L1
    done
done



# #!/bin/bash

# # Usage on Slurm:
# # sbatch repro_scripts/convalibi_dilated.sh AppliancesEnergy
# # Output will appear in a file 'slurm-N.out' where N is the job ID.
# # Change dataset in DATA below.

# # Request the full partition
# #SBATCH -p full
# #SBATCH --exclude=c0020,c0002

# # Name the job so it's meaningful in the job list
# #SBATCH -J erpe_convalibi
# # Request 1 V100 GPU
# #SBATCH --gpus v100:1
# # Request 4 CPU cores (8 hyperthreads).
# #SBATCH -c 8
# # Specify the resources should be assigned to a single task on one node.
# #SBATCH -N 1 -n 1
# # Request a total of 80GB RAM
# #SBATCH --mem=20GB
# # Request a walltime limit of 72 hours
# #SBATCH -t 72:00:00

# # Activate environment
# source ~/.bashrc
# conda activate tser

# # Dataset
# # AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent 
# DATA="BenzeneConcentration"

# # Batch size: 16 for AppliancesEnergy, otherwise 128
# if [ "$DATA" = "AppliancesEnergy" ]; then
#     BS=16
#     PATIENCE=500
# elif [ "$DATA" = "BenzeneConcentration" ]; then
#     BS=128
#     PATIENCE=500
# else
#     BS=128
#     PATIENCE=100
# fi

# if [ "$DATA" = "IEEEPPG" ]; then
#     PATCH=8
#     STRIDE=4
# elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
#     PATCH=4
#     STRIDE=4
# else
#     PATCH=1
#     STRIDE=1
# fi

# if [ "$DATA" = "AppliancesEnergy" ] || [ "$DATA" = "IEEEPPG" ]; then
#     LR=1e-2
# else
#     LR=1e-3
# fi


# # OUTPUT FILE
# OUTPUT_FILE="${DATA}_SENS_SMOOTH"

# # TUNING
# for CONVIT_SLOPE in 0.25
# do
#     for HEADS in 16
#     do
#         for LAM in 1e-2 1e-1
#         do
#             for LAM2 in 1e-3
#             do
#                 for SEED in 0 1 2
#                 do
#                     # BASIC
#                     PARAM_STR="LR=${LR}_POOLSMOOTH=${LAM}_L1=${LAM2}_SLOPE=${CONVIT_SLOPE}_HEADS=${HEADS}_SEED=${SEED}"
#                     python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
#                         --records_file "output/${OUTPUT_FILE}.xls" \
#                         --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
#                         --pattern TRAIN --val_pattern TEST \
#                         --epochs 2000 --patience $PATIENCE \
#                         --lr $LR --batch_size $BS \
#                         --num_layers 3 --num_heads $HEADS --d_model 128 --dim_feedforward 256 \
#                         --optimizer RAdam --task regression \
#                         --plot_loss --plot_accuracy \
#                         --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
#                         --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat \
#                         --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos --convit_slope $CONVIT_SLOPE \
#                         --pool seqpool_multihead --reg_lambda_pool $LAM --reg_lambda $LAM --l1_reg $LAM2
#                 done
#             done
#         done
#     done
# done
