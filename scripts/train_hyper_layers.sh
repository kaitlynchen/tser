data_dir=$1

for SEED in 0 1 2 3 4
do
    for num_layers in 1 3 5 7
    do
        data=$(echo $data_dir | cut -d "/" -f 2)
        python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX with attention smoothing on ${data}, hyperparameter search on layers" \
                --seed $SEED --name ClimaX_smooth_${data}_num_layers_${num_layers} --records_file climax_smooth_${data}_hyperparameter_num_layers.xls \
                --data_dir $data_dir --data_class tsra --pattern TRAIN --test_pattern TEST --val_ratio 0.2 --epochs 200 --lr 0.001 \
                --num_layers $num_layers --num_heads 16 --dim_feedforward 256 --optimizer RAdam --batch_size 128 --pos_encoding learnable --d_model 128 \
                --task regression --normalization standardization --model climax_smooth --patch_length 16 --stride 8 --reg_lambda 0.01 --smooth_attention
    done
done