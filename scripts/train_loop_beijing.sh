for SEED in 0
do
    # ERPE INITIALIZATIONS
	# erpe conv-alibi init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
            --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # erpe zero init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe zero init, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_zero_init_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_zero_init_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_zero_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # erpe alibi init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe alibi init, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_alibi_init_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_alibi_init_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_alibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # erpe convit init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convit init, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convit_init_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convit_init_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convit_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # convit, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, convit, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_convit_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_convit_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding convit --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # WHERE TO ADD RELPOS
    # erpe convalibi init, after gating, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, after gating, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_after_gating_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_after_gating_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos after_gating \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # erpe convalibi init, after, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, after, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_after_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_after_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos after \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # erpe convalibi init, before, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, before, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_before_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_before_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos before \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # erpe convalibi init, no rel pos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, no relpos, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_no_relpos_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_no_relpos_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding none \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # ABS POS ENCODING
    # erpe convalibi init, only relpos, learnable sin init added at start
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, only relpos, learnable sin init added at start, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_learnable_sin_init_add_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_learnable_sin_init_add_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos start_add --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3

    # erpe convalibi init, only relpos, learnable uniform init added at start
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, only relpos, learnable uniform init added at start, BeijingPM10Quality" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_learnable_uniform_init_add_beijing_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_learnable_uniform_init_add_beijing.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BeijingPM10Quality --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_uniform_init --where_to_add_abspos start_add --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead --convit_slope 0.25 --reg_lambda_pool 1e-3 --reg_lambda 1e-3
done