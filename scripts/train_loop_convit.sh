for SEED in 0 1 2 3 4
do
    python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX and ConViT hybrid, no GPSA layers, trained from scratch, BenzeneConcentration" \
            --seed $SEED --name ClimaX_ConViT_seed_${SEED}_LR_${LR}_verification --records_file climax_convit_verification.xls --data_dir data/BenzeneConcentration --data_class tsra --pattern TRAIN \
            --val_pattern TEST --epochs 200 --lr 0.01 --num_heads 8 --dim_feedforward 512 --optimizer RAdam --batch_size 128 --num_layers 8 --num_gpsa_layers 0 \
            --pos_encoding learnable --d_model 128 --task regression --normalization standardization --model convit --patch_length 16 --stride 8 
done