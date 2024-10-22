# for SEED in 0 1 2 3 4
# do
#     for LAYERS in 5 7 9
#     do
#         python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX and ConViT hybrid, trained from scratch, BenzeneConcentration" \
#                 --seed $SEED --name ConViT_BenzeneConcentration_seed_${SEED}_${LAYERS}_encoder_blocks_stride_2_no_gpsa --records_file convit_no_GPSA_stride_2_benzene.xls \
#                 --data_dir data/BenzeneConcentration --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 200 --lr 0.0001 --num_heads 3 \
#                 --dim_feedforward 256 --optimizer RAdam --batch_size 128 --num_layers $LAYERS --num_gpsa_layers 0 --pos_encoding learnable --d_model 144 --task regression \
#                 --normalization standardization --model convit_ --patch_length 8 --stride 2 
#     done
# done

for SEED in 0 1 2 3 4
do
    for LAYERS in 5 7 9
    do
        for SMOOTH in 1e-3 1e-2 1e-1
        do 
            python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX and ConViT hybrid, with attention smoothing, no GPSA layers, trained from scratch, BenzeneConcentration" \
                    --seed $SEED --name ConViT_attention_smoothing_no_gpsa_layers_BenzeneConcentration_seed_${SEED}_${LAYERS}_encoder_blocks_smoothing_lambda_${SMOOTH} --records_file convit_no_GPSA_attention_smoothing_stride_2_benzene.xls \
                    --data_dir data/BenzeneConcentration --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 200 --lr 0.0001 --num_heads 3 \
                    --dim_feedforward 256 --optimizer RAdam --batch_size 128 --num_layers $LAYERS --num_gpsa_layers 0 --pos_encoding learnable --d_model 144 --task regression \
                    --normalization standardization --model convit_smooth --patch_length 8 --stride 2 --reg_lambda $SMOOTH --smooth_attention
        done
    done
done