for SEED in 0 1 2
do
    # for LR in 1e-3 3e-4 1e-4 3e-5 1e-5
    # do
    python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX with original hyperparameters on BenzeneConcentration, seqpool final layer" \
            --seed $SEED --name ClimaX_smooth_BenzeneConcentration_original_hyperparameters_seqpool_seed_${SEED} \
            --records_file climax_smooth_benzene_original_seqpool.xls --data_dir data/BenzeneConcentration --data_class tsra --pattern TRAIN \
            --val_pattern TEST --epochs 200 --lr 0.001 --num_heads 8 --dim_feedforward 256 --optimizer RAdam --batch_size 128 \
            --pos_encoding learnable --d_model 128 --task regression --normalization standardization --model climax_smooth --patch_length 1 \
            --stride 1 --reg_lambda 0.01 --smooth_attention
    # done
done