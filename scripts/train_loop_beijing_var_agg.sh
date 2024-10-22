for SEED in 0 1 2 3 4
do
    python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX with attention smoothing and variable aggregation on BeijingPM25Quality, testing" \
                            --seed $SEED --name ClimaX_smooth_BeijingPM25Quality_var_agg_seed_${SEED}_test \
                            --records_file climax_smooth_beijing_hyperparameter_var_agg_test.xls --data_dir data/BeijingPM25Quality --data_class tsra --pattern TRAIN \
                            --val_pattern TEST --epochs 200 --lr 0.001 --num_layers 3 --num_heads 16 --dim_feedforward 256 --optimizer RAdam --batch_size 64 \
                            --pos_encoding learnable --d_model 128 --task regression --normalization standardization --model climax_smooth --patch_length 8 \
                            --stride 2 --reg_lambda 0.1 --smooth_attention --agg_vars
done