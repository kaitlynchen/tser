for SEED in 0 1 2
do
    python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX with original Zerveas hyperparameters on validation set, AppliancesEnergy" \
            --seed $SEED --name ClimaX_original_hyperparameters_AppliancesEnergy_seed_${SEED} \
            --records_file climax_appliances_smooth_seqpool_val_set.xls --data_dir data/AppliancesEnergy --data_class tsra --pattern TRAIN \
            --val_ratio 0.2 --epochs 500 --lr 0.001 --num_heads 8 --dim_feedforward 512 --optimizer RAdam --batch_size 128 \
            --pos_encoding learnable --d_model 128 --num_layers 1 --task regression --normalization standardization --model climax_smooth_pool --patch_length 1 \
            --stride 1 --normalize_label --smooth_attention
done