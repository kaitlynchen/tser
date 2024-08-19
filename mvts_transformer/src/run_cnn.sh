# Baseline
DATA=${1}

for LR in 1e-3 1e-2
do
    for POOL in seqpool
    do
        for CONV_TYPE in local
        do
            for SEED in 0 1 2
            do
                python main.py --comment "CNN_${DATA}" \
                    --seed $SEED --name CNN_${DATA} \
                    --records_file stats/CNN_${DATA}_${POOL}_${CONV}.xls \
                    --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                    --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr $LR \
                    --optimizer RAdam --task regression --model local_cnn --pool $POOL --conv_type $CONV_TYPE \
                    --plot_loss --plot_accuracy
            done
        done
    done
done
