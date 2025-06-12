for SEED in 0 1 2
do
    python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX without original Zerveas hyperparameters, AppliancesEnergy" \
            --seed $SEED --name ClimaX_original_hyperparameters_AppliancesEnergy_seed_${SEED} \
            --records_file climax_appliances_original_hyperparameters.xls --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra --pattern TRAIN \
            --val_pattern TEST --epochs 500 --lr 0.001 --num_heads 8 --dim_feedforward 512 --optimizer RAdam --batch_size 128 \
            --pos_encoding learnable --d_model 128 --num_layers 3 --task regression --normalization standardization --model climax --patch_length 1 \
            --stride 1
done