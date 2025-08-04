#!/bin/bash

# Usage on Slurm:
# sbatch repro_scripts/convalibi_dilated.sh AppliancesEnergy
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

# Dataset
# AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent 
DATA="BenzeneConcentration"

# Batch size: 16 for AppliancesEnergy, otherwise 128
if [ "$DATA" = "AppliancesEnergy" ]; then
    BS=16
else
    BS=128
fi
BS=11
if [ "$DATA" = "IEEEPPG" ]; then
    PATCH=8
    STRIDE=4
elif [ "$DATA" = "LiveFuelMoistureContent" ]; then
    PATCH=4
    STRIDE=4
else
    PATCH=1
    STRIDE=1
fi

# REPRO CLIMAX + LOCAL MASK
for MASK in -1 0 2 4
do
    for POOL in seqpool_multihead
    do
        OUTPUT_FILE="${DATA}_TST_CLIMAX_FIXEDMASK=${MASK}_POOL=${POOL}"

        for LR in 1e-2 1e-3
        do
            for SEED in 0
            do
                # BASIC
                PARAM_STR="LR=${LR}_SEED=${SEED}"
                python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
                    --records_file "output/${OUTPUT_FILE}.xls" \
                    --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                    --pattern TRAIN --val_ratio 0.2 \
                    --epochs 2000 --patience 500 \
                    --lr $LR --batch_size $BS \
                    --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                    --optimizer RAdam --task regression \
                    --plot_loss --plot_accuracy \
                    --model climax_smooth --patch_length $PATCH --stride $STRIDE --smooth_attention --normalize_label \
                    --pos_encoding learnable_uniform_init --where_to_add_abspos start_add \
                    --pool $POOL --local_mask $MASK
            done
        done

        # TEST
        python test_best.py --records_file "output/${OUTPUT_FILE}.xls"
    done
done