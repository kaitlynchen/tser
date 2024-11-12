# Baseline
DATA=${1}

for LR in 1e-2
do
    for POOL in seqpool_multihead
    do
        for CONV_TYPE in local
        do
            for SEED in 0 1 2
            do
                python main.py --comment "CNN_MLP_STEP_${DATA}" \
                    --seed $SEED --name CNN_MLP_STEP_${DATA} \
                    --records_file CNN_MLP_STEP_${DATA}_${POOL}_${CONV}.xls \
                    --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                    --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr $LR \
                    --optimizer RAdam --task regression --normalize_label \
                    --model local_cnn --pool $POOL --conv_type $CONV_TYPE --pos_encoding learnable_sin_init  \
                    --plot_loss --plot_accuracy --lr_step "200,300,400" --lr_factor 0.2
            done
        done
    done
done
