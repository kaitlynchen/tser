for SEED in 0 1 2 3 4
do
    for LAYERS in 5 7 9
    do
        python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX without attention smoothing, trained from scratch, BenzeneConcentration" \
                --seed $SEED --name ClimaX_BenzeneConcentration_seed_${SEED}_${LAYERS}_encoder_blocks_stride_2 --records_file climax_stride_2_benzene.xls \
                --data_dir data/BenzeneConcentration --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 200 --lr 0.0001 --num_heads 3 \
                --dim_feedforward 256 --optimizer RAdam --batch_size 128 --num_layers $LAYERS --pos_encoding learnable --d_model 144 --task regression \
                --normalization standardization --model climax --patch_length 8 --stride 2 
    done
done