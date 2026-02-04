DATASET=$1
for PATCH_LENGTH in 1 4 8 16
do
    for STRIDE_FACTOR in 0.5 1
    do 
        for lr in 1e-3 1e-2 1e-1
        do 
            for smooth_factor in 0 1e-3 1e-2
            do
                stride=$(echo $STRIDE_FACTOR*$PATCH_LENGTH | bc)
                stride=$(printf "%.f" $stride)
                data=$(echo $DATASET | cut -d "/" -f 2)
                python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX with attention smoothing on ${data}, hyperparameter search on patch size and stride" \
                        --seed 0 --name ClimaX_smooth_${data}_patch_${PATCH_LENGTH}_stride_${STRIDE}_lr_${lr}_smooth_${smooth} \
                        --records_file climax_smooth_${data}_hyperparameter_patch_stride.xls --data_dir ${DATASET} --data_class tsra --pattern TRAIN \
                        --test_pattern TEST --val_ratio 0.2 --epochs 200 --lr $lr --num_layers 3 --num_heads 16 --dim_feedforward 256 --optimizer RAdam --batch_size 128 \
                        --pos_encoding learnable --d_model 128 --task regression --normalization standardization --model climax_smooth --patch_length $PATCH_LENGTH \
                        --stride $stride --reg_lambda $smooth_factor --smooth_attention
            done
        done
    done
done