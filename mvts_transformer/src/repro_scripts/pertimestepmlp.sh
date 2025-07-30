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

# Dataset
# AppliancesEnergy BeijingPM10Quality BeijingPM25Quality BenzeneConcentration IEEEPPG LiveFuelMoistureContent 
DATA="IEEEPPG"

# Batch size: 16 for AppliancesEnergy, otherwise 128
if [ "$DATA" = "AppliancesEnergy" ]; then
    BS=16
else
    BS=128
fi
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

# Run both per-timestep MLP and LSTM
for CONV_TYPE in per_timestep
do
    OUTPUT_FILE="${DATA}_LOCAL42_${CONV_TYPE}"
    for POOL in seqpool_multihead
    do
        for WHERE_ABSPOS in before_pool_concat
        do
            for LR in 1e-3 1e-2
            do
                for LAM in 0
                do
                    for SEED in 0
                    do
                        PARAM_STR="LR=${LR}_POOLSMOOTH=${LAM}_SEED=${SEED}"
                        python main.py --seed $SEED --name "${OUTPUT_FILE}_${PARAM_STR}" \
                            --records_file "output/${OUTPUT_FILE}.xls" \
                            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                            --pattern TRAIN --val_ratio 0.2 \
                            --epochs 2000 --patience 200 \
                            --lr $LR --batch_size $BS \
                            --num_heads 16 --d_model 128 \
                            --optimizer RAdam --task regression --normalize_label \
                            --model local_cnn --conv_type $CONV_TYPE \
                            --patch_length $PATCH --stride $STRIDE --smooth_attention \
                            --pos_encoding learnable_sin_init --where_to_add_abspos $WHERE_ABSPOS \
                            --pool $POOL --reg_lambda_pool $LAM \
                            --plot_loss --plot_accuracy
                    done
                done
            done
        done
    done

    python test_best.py --records_file "output/${OUTPUT_FILE}.xls"
done