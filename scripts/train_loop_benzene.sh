for SEED in 0
do
    # ERPE INITIALIZATIONS
	# erpe conv-alibi init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
            --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe zero init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe zero init, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_zero_init_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_zero_init_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_zero_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe alibi init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe alibi init, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_alibi_init_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_alibi_init_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_alibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convit init, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convit init, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convit_init_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convit_init_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convit_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # convit, only relpos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, convit, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_convit_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_convit_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding convit --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # WHERE TO ADD RELPOS
    # erpe convalibi init, after gating, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, after gating, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_after_gating_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_after_gating_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos after_gating \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convalibi init, after, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, after, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_after_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_after_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos after \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convalibi init, before, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, before, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_before_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_before_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos before \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convalibi init, no rel pos, learnable sin init concatenated before pool
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, no relpos, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_no_relpos_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_no_relpos_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding none \
                --pos_encoding learnable_sin_init --where_to_add_abspos before_pool_concat --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # ABS POS ENCODING
    # erpe convalibi init, only relpos, learnable sin init added at start
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, only relpos, learnable sin init added at start, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_learnable_sin_init_add_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_learnable_sin_init_add_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_sin_init --where_to_add_abspos start_add --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 

    # erpe convalibi init, only relpos, learnable uniform init added at start
	python mvts_transformer/src/main.py --comment "ClimaX with multihead seqpool, erpe convalibi init, only relpos, learnable uniform init added at start, BenzeneConcentration" \
                --seed $SEED --name ClimaX_multihead_seqpool_erpe_convalibi_init_learnable_uniform_init_add_benzene_seed_${SEED} --output_dir ./output \
                --records_file climax_multihead_seqpool_erpe_convalibi_init_learnable_uniform_init_add_benzene.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/BenzeneConcentration --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 2000 --patience 200 --batch_size 128 --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --task regression --model climax_smooth --relative_pos_encoding erpe_convalibi_init --where_to_add_relpos only_relpos \
                --pos_encoding learnable_uniform_init --where_to_add_abspos start_add --smooth_attention --plot_loss \
                --plot_accuracy --normalize_label --pool seqpool_multihead 
done