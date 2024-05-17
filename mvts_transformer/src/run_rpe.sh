
for DATA in AppliancesEnergy BenzeneConcentration BeijingPM10Quality BeijingPM25Quality IEEEPPG LiveFuelMoistureContent
do
    for POSENC in none erpe alibi
    do
        # Baseline
        for SEED in 0 1 2
        do
            # Usage: ./run.sh AppliancesEnergy, from mvts_transformer/src directory. Run "mkdir output" first.
            python main.py --comment "ClimaX BASELINE on ${DATA}" \
                --seed $SEED --name ClimaX_smooth_${DATA}_patch=1_${POSENC}_seed=${SEED} \
                --records_file climax_smooth_${DATA}_hyperparameter_test.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr 0.001 \
                --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 512 \
                --optimizer RAdam  --pos_encoding learnable --relative_pos_encoding $POSENC --task regression \
                --model climax_smooth --patch_length 1 --stride 1 --reg_lambda 0 --smooth_attention

            python main.py --comment "ClimaX PATCH8/2 on ${DATA}" \
                --seed $SEED --name ClimaX_smooth_${DATA}_patch=8_${POSENC}_seed=${SEED} \
                --records_file climax_smooth_${DATA}_hyperparameter_test.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr 0.001 \
                --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 512 \
                --optimizer RAdam  --pos_encoding learnable --relative_pos_encoding $POSENC --task regression \
                --model climax_smooth --patch_length 8 --stride 2 --reg_lambda 0 --smooth_attention
        done
    done
done

# Note: --pattern TRAIN --val_ratio 0.2 validates on 20% of train set
# Note: --pattern TRAIN --val_pattern TEST validates on test set
