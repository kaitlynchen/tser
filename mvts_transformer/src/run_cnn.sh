# CNN baseline. Usage:
# ./run_cnn.sh AppliancesEnergy
# # Josh's results:
# 1) CNN_AppliancesEnergy_seqpool_multihead_local_VAL.xls: 3.22 (lr=1e-3), 3.68 (lr=1e-2)
# 2) CNN_AppliancesEnergy_seqpool_multihead_local_TEST.xls: 2.26 (lr=1e-3), 2.13 (lr=1e-2)
# 3) CNN_AppliancesEnergy_seqpool_multihead_per_timestep_VAL.xls: 2.79 (lr=1e-2)
# 4) CNN_AppliancesEnergy_seqpool_multihead_per_timestep_TEST.xls: 2.05 (lr=1e-2)

# DATA=${1}

for DATA in AppliancesEnergy BenzeneConcentration BeijingPM10Quality BeijingPM25Quality IEEEPPG LiveFuelMoistureContent
do
    for LR in 1e-3
    do
        for POOL in seqpool_multihead
        do
            for CONV_TYPE in per_timestep
            do
                for SEED in 0 1 2
                do

                    python main.py --comment "CNN_${DATA}_${POOL}_${CONV_TYPE}_VAL" \
                        --seed $SEED --name CNN_${DATA} \
                        --records_file CNN3_${DATA}_${POOL}_${CONV_TYPE}_VAL.xls \
                        --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                        --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr $LR \
                        --optimizer RAdam --task regression --normalize_label \
                        --model local_cnn --pool $POOL --conv_type $CONV_TYPE --pos_encoding learnable_sin_init  \
                        --plot_loss --plot_accuracy

                    python main.py --comment "CNN_${DATA}_${POOL}_${CONV_TYPE}_TEST" \
                        --seed $SEED --name CNN_${DATA} \
                        --records_file CNN3_${DATA}_${POOL}_${CONV_TYPE}_TEST.xls \
                        --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                        --pattern TRAIN --val_pattern TEST --epochs 500 --lr $LR \
                        --optimizer RAdam --task regression --normalize_label \
                        --model local_cnn --pool $POOL --conv_type $CONV_TYPE --pos_encoding learnable_sin_init  \
                        --plot_loss --plot_accuracy
                done
            done
        done
    done
done