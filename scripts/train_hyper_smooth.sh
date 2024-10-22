data_dir=$1
patch=$2
stride=$3

for SEED in 0 1 2
do
    for smooth_lambda in 0 0.1 1 10
    do
        data=$(echo $data_dir | cut -d "/" -f 2)
        python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX with attention smoothing on ${data}, hyperparameter search on smoothing factor" \
                --seed $SEED --name ClimaX_smooth_${data}_smooth_lambda_${smooth_lambda}_patch_${patch}_stride_${stride} --records_file climax_smooth_${data}_patch_${patch}_stride_${stride}_hyperparameter_smooth_lambda.xls \
                --data_dir $data_dir --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 500 --lr 0.001 \
                --num_layers 3 --num_heads 16 --dim_feedforward 256 --optimizer RAdam --batch_size 128 --pos_encoding learnable --d_model 128 \
                --task regression --normalization standardization --model climax_smooth --patch_length $patch --stride $stride --reg_lambda 0.01 --smooth_attention
    done
done