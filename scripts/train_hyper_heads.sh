data_dir=$1

for SEED in 0 1 2 3 4
do
    for heads in 2 4 8 16
    do
        for d_model in 128 144
        do
            data=$(echo $data_dir | cut -d "/" -f 2)
            python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX with attention smoothing on ${data}, hyperparameter search on num_heads" \
                    --seed $SEED --name ClimaX_smooth_${data}_heads_${heads}_d_model_${d_model} --records_file climax_smooth_${data}_hyperparameter_num_heads.xls \
                    --data_dir $data_dir --data_class tsra --pattern TRAIN --test_pattern TEST --val_ratio 0.2 --epochs 200 --lr 0.001 \
                    --num_layers 3 --num_heads $heads --dim_feedforward 256 --optimizer RAdam --batch_size 128 --pos_encoding learnable --d_model $d_model \
                    --task regression --normalization standardization --model climax_smooth --patch_length 16 --stride 8 --reg_lambda 0.01 --smooth_attention
        done
    done
done