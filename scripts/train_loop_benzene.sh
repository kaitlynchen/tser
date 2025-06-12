reg_lambda=$1

for SEED in 0 1 2
do
    python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX with attention smoothing on BenzeneConcentration, testing" \
                            --seed $SEED --name ClimaX_smooth_BenzeneConcentration_seed_${SEED}_test \
                            --records_file climax_smooth_benzene_smoothing_factor_${reg_lambda}_hyperparameter_test.xls --data_dir data/BenzeneConcentration --data_class tsra --pattern TRAIN \
                            --val_pattern TEST --epochs 200 --lr 0.001 --num_layers 3 --num_heads 3 --dim_feedforward 256 --optimizer RAdam --batch_size 128 \
                            --pos_encoding learnable --d_model 144 --task regression --normalization standardization --model climax_smooth --patch_length 4 \
                            --stride 2 --reg_lambda $reg_lambda --smooth_attention
done