# Best hyperparams for AppliancesEnergy
# Note 2000 epochs are needed.
# Run `mkdir output_repro` first.
for DATA in AppliancesEnergy
do
    for SEED in 0 1 2
    do
        # Original Zerveas TST: choosing model by test loss
        python main.py --comment "TST BASELINE on ${DATA}" \
            --seed $SEED --name REPROTEST_MVTS_${DATA}_TEST \
            --records_file output_repro/REPROTEST_MVTS_${DATA}_TEST.xls \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --pattern TRAIN --val_pattern TEST --epochs 2000 --lr 0.001 --batch_size 128 \
            --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
            --optimizer RAdam --pos_encoding learnable --task regression \
            --model transformer --plot_loss --plot_accuracy

        # Climax-Smooth: chooosing model by test loss. Should give same result as second command.
        python main.py --comment "ClimaX on ${DATA}" \
            --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST \
            --records_file output_repro/REPROTEST_CLIMAX_${DATA}_TEST.xls \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --pattern TRAIN --val_pattern TEST --epochs 2000 --lr 0.001 --batch_size 128 \
            --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
            --optimizer RAdam --pos_encoding learnable --task regression \
            --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
            --plot_loss --plot_accuracy
            # --relative_pos_encoding erpe --where_to_add_relpos after

        # # Same as above but with --normalize_label
        # python main.py --comment "ClimaX on ${DATA}" \
        #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST_NORMALIZELABEL \
        #     --records_file REPROTEST_CLIMAX_${DATA}_TEST.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_pattern TEST --epochs 2000 --lr 0.001 --batch_size 128 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
        #     --optimizer RAdam --pos_encoding learnable --task regression \
        #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
        #     --plot_loss --plot_accuracy --normalize_label

        # # Original Zerveas TST: choosing model by val loss
        # python main.py --comment "TST BASELINE on ${DATA}" \
        #     --seed $SEED --name REPROTEST_MVTS_${DATA}_VAL \
        #     --records_file output_repro/REPROTEST_MVTS_${DATA}_VAL.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_ratio 0.2 --epochs 1500 --lr 0.001 --batch_size 128 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
        #     --optimizer RAdam --pos_encoding learnable --task regression \
        #     --model transformer --plot_loss --plot_accuracy

        # # Climax-Smooth: chooosing model by val loss. Should give same result as first command.
        # python main.py --comment "ClimaX on ${DATA}" \
        #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_VAL \
        #     --records_file REPROTEST_CLIMAX_${DATA}_VAL.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr 0.001 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
        #     --optimizer RAdam --pos_encoding learnable --task regression \
        #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
        #     --plot_loss --plot_accuracy
    done
done
