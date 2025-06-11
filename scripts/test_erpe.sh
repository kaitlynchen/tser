for SEED in 0 1 2
do
    for LR in 1e-2 1e-3 1e-4
    do
        # AppliancesEnergy: climax smooth with multihead seqpool, learnable absolute pos encoding before pool layer, and erpe
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and learnable absolute pos encoding before pool layer, AppliancesEnergy" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_learnable_before_pool_AppliancesEnergy_seed_${SEED} --output ./erpe-output \
                --records_file climax_appliances_multihead_seqpool_erpe_convit_init_learnable_before_pool.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 --pos_encoding learnable \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool seqpool_multihead

        # AppliancesEnergy: climax smooth with multihead seqpool, learnable absolute pos encoding at pool layer, and erpe
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and learnable absolute pos encoding only at pool layer, AppliancesEnergy" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_learnable_at_pool_AppliancesEnergy_seed_${SEED} --output ./erpe-output \
                --records_file climax_appliances_multihead_seqpool_erpe_convit_init_learnable_at_pool.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool learnable_seqpool_multihead

        # AppliancesEnergy: climax smooth with multihead seqpool, no absolute pos encoding, and erpe
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and no absolute pos encoding, AppliancesEnergy" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_no_abs_pos_encoding_AppliancesEnergy_seed_${SEED} --output ./erpe-output \
                --records_file climax_appliances_multihead_seqpool_erpe_no_abs_pos_encoding_test.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 --pos_encoding none \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool seqpool_multihead

        # AppliancesEnergy: climax smooth with multihead seqpool, absolute pos encoding at seqpool layer, and erpe
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and learnable absolute pos encoding only at pool layer, AppliancesEnergy" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_abs_pos_encoding_at_seqpool_AppliancesEnergy_seed_${SEED} --output ./erpe-output \
                --records_file climax_appliances_multihead_seqpool_erpe_convit_init_abs_pos_encoding_at_seqpool.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 --pos_encoding none \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool seqpool_multihead_posenc

        # AppliancesEnergy: climax smooth with multihead seqpool, learnable absolute pos encoding at final seqpool layer, and erpe
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and learnable absolute pos encoding at final seqpool layer, AppliancesEnergy" \
                --seed $SEED --name ClimaX_final_multihead_seqpool_pos_encoding_erpe_AppliancesEnergy_seed_${SEED} \
                --records_file climax_appliances_final_multihead_seqpool_erpe.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 50 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 --pos_encoding learnable \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool final_seqpool_multihead_posenc


        # ---------------
        # ERPE ALIBI INIT
        # ---------------
        # AppliancesEnergy: climax smooth with multihead seqpool, learnable absolute pos encoding before pool layer, and erpe alibi init
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and learnable absolute pos encoding before pool layer, AppliancesEnergy" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_alibi_learnable_before_pool_AppliancesEnergy_seed_${SEED} \
                --records_file climax_appliances_multihead_seqpool_erpe_alibi_init_learnable_before_pool.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 --pos_encoding learnable \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe_alibi_init \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool seqpool_multihead

        # AppliancesEnergy: climax smooth with multihead seqpool, learnable absolute pos encoding at pool layer, and erpe alibi init
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and learnable absolute pos encoding only at pool layer, AppliancesEnergy" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_alibi_learnable_at_pool_AppliancesEnergy_seed_${SEED} \
                --records_file climax_appliances_multihead_seqpool_erpe_alibi_init_learnable_at_pool.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe_alibi_init \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool learnable_seqpool_multihead

        # AppliancesEnergy: climax smooth with multihead seqpool, no absolute pos encoding, and erpe alibi init
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and learnable absolute pos encoding only at pool layer, AppliancesEnergy" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_alibi_no_abs_pos_encoding_AppliancesEnergy_seed_${SEED} \
                --records_file climax_appliances_multihead_seqpool_erpe_alibi_init_no_abs_pos_encoding.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 --pos_encoding none \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe_alibi_init \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool seqpool_multihead

        # AppliancesEnergy: climax smooth with multihead seqpool, absolute pos encoding at seqpool layer, and erpe alibi init
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and learnable absolute pos encoding only at pool layer, AppliancesEnergy" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_alibi_abs_pos_encoding_at_seqpool_AppliancesEnergy_seed_${SEED} \
                --records_file climax_appliances_multihead_seqpool_erpe_alibi_init_abs_pos_encoding_at_seqpool.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 --pos_encoding none \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe_alibi_init \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool seqpool_multihead_posenc

        # AppliancesEnergy: climax smooth with multihead seqpool, learnable absolute pos encoding at final seqpool layer, and erpe alibi init
        python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe, and learnable absolute pos encoding at final seqpool layer, AppliancesEnergy" \
                --seed $SEED --name ClimaX_final_multihead_seqpool_pos_encoding_erpe_AppliancesEnergy_seed_${SEED} \
                --records_file climax_appliances_final_multihead_seqpool_erpe_alibi_init.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/AppliancesEnergy --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 50 --lr $LR --batch_size 16 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 --pos_encoding learnable \
                --optimizer RAdam --task regression --model climax_smooth --patch_length 1 --stride 1 --relative_pos_encoding erpe_alibi_init \
                --where_to_add_relpos only_relpos --smooth_attention --plot_loss --plot_accuracy --normalize_label --pool final_seqpool_multihead_posenc
    done
done