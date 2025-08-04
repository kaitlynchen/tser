for SEED in 0
do
    # ERPE INITIALIZATIONS
	# erpe conv-alibi init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
            --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe zero init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe zero init, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_zero_init_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_zero_init_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_zero_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe alibi init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe alibi init, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_alibi_init_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_alibi_init_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_alibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convit init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convit init, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convit_init_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convit_init_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convit_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # convit, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, convit, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_convit_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_convit_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding convit --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # WHERE TO ADD RELPOS
    # erpe convalibi init, after gating, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, after gating, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_after_gating_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_after_gating_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos after_gating \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convalibi init, after, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, after, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_after_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_after_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos after \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convalibi init, before, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, before, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_before_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_before_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos before \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convalibi init, no rel pos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, no relpos, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_no_relpos_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_no_relpos_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding none \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # ABS POS ENCODING
    # erpe convalibi init, only relpos, learnable sin init added at start
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, only relpos, learnable sin init added at start, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_learnable_sin_init_add_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_learnable_sin_init_add_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos start_add --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convalibi init, only relpos, learnable uniform init added at start
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, only relpos, learnable uniform init added at start, LiveFuelMoistureContent" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_learnable_uniform_init_add_live_fuel_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_learnable_uniform_init_add_live_fuel.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/LiveFuelMoistureContent --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --num_layers 3 --num_heads 8 --d_model 64 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_uniform_init --where_to_add_abspos start_add --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 
done