for SEED in 0 1 2
do
    # for PATCH_LENGTH in 4 8 12 16 20
    # do
    #     for STRIDE_FACTOR in 0.25 0.5 0.75 1
    #     do 
    #         stride=$(echo $STRIDE_FACTOR*$PATCH_LENGTH | bc)
    #         stride=$(printf "%.f" $stride)
    python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX without original Zerveas hyperparameters, BenzeneConcentration" \
            --seed $SEED --name ClimaX_original_hyperparameters_BenzeneConcentration_seed_${SEED} \
            --records_file climax_benzene_original_hyperparameters.xls --data_dir data/BenzeneConcentration --data_class tsra --pattern TRAIN \
            --val_pattern TEST --epochs 200 --lr 0.001 --num_heads 8 --dim_feedforward 256 --optimizer RAdam --batch_size 128 \
            --pos_encoding learnable --d_model 128 --num_layers 1 --task regression --normalization standardization --model climax --patch_length 1 \
            --stride 1
    #     done
    # done
done