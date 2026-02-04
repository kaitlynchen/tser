data_dir=$1

for SEED in 0 1 2 3 4
do
    for dim_ffw in 64 128 256 512
    do
        data=$(echo $data_dir | cut -d "/" -f 2)
        python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX with attention smoothing on ${data}, hyperparameter search on dim_ffw" \
                --seed $SEED --name ClimaX_smooth_${data}_dim_ffw_${dim_ffw} --records_file climax_smooth_${data}_hyperparameter_dim_ffw.xls \
                --data_dir $data_dir --data_class tsra --pattern TRAIN --test_pattern TEST --val_ratio 0.2 --epochs 200 --lr 0.001 \
                --num_layers 3 --num_heads 16 --dim_feedforward $dim_ffw --optimizer RAdam --batch_size 128 --pos_encoding learnable --d_model 128 \
                --task regression --normalization standardization --model climax_smooth --patch_length 16 --stride 8 --reg_lambda 0.01 --smooth_attention
    done
done