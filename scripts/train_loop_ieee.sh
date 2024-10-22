for SEED in 0 1 2 3 4
do
    for LR in 1e-3 3e-4 1e-4 3e-5 1e-5
    do  
        python mvts_transformer/src/main.py --output_dir output/test --comment "ClimaX and ConViT hybrid, trained from scratch, IEEEPPG" \
                --seed $SEED --name ClimaX_ConViT_IEEEPPG_seed_${SEED}_LR_$LR --records_file climax_convit_results_ieeeppg.xls --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/IEEEPPG --data_class tsra --pattern TRAIN \
                --val_pattern TEST --epochs 200 --lr $LR --num_heads 3 --dim_feedforward 256 --num_layers 12 --num_gpsa_layers 10 --optimizer RAdam --batch_size 128 \
                --pos_encoding learnable --d_model 288 --task regression --normalization standardization --model convit --patch_length 8 --stride 4 
    done
done