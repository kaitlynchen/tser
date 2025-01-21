
# DATA=${1}

# ./run_repro.sh AppliancesEnergy
# Josh's results:
# 1) REPROTEST_MVTS_AppliancesEnergy_VAL.xls: 3.12
# 2) REPROTEST_MVTS_AppliancesEnergy_TEST.xls: 2.72
# 3) REPROTEST_CLIMAX_AppliancesEnergy_VAL.xls: 3.14
# 4) REPROTEST_CLIMAX_AppliancesEnergy_TEST.xls: 2.56
# 5) REPROTEST_CLIMAX_AppliancesEnergy_TEST.xls, normalize_label: 2.60

# ./run_repro.sh BeijingPM10Quality
# Zerveas reported 91.344

#for DATA in AppliancesEnergy BenzeneConcentration BeijingPM10Quality BeijingPM25Quality IEEEPPG LiveFuelMoistureContent
#do
for DATA in BenzeneConcentration
do
    for SEED in 0 1 2
    do
        # Original Zerveas TST: choosing model by val loss
        python main.py --comment "TST BASELINE on ${DATA}" \
            --seed $SEED --name REPROTEST_MVTS_${DATA}_VAL \
            --records_file REPROTEST_MVTS_${DATA}_VAL.xls \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr 0.001 --batch_size 128 \
            --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
            --optimizer RAdam --pos_encoding learnable --task regression \
            --model transformer --plot_loss --plot_accuracy

        # Original Zerveas TST: choosing model by test loss
        python main.py --comment "TST BASELINE on ${DATA}" \
            --seed $SEED --name REPROTEST_MVTS_${DATA}_TEST \
            --records_file REPROTEST_MVTS_${DATA}_TEST.xls \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --pattern TRAIN --val_pattern TEST --epochs 500 --lr 0.001 --batch_size 128 \
            --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
            --optimizer RAdam --pos_encoding learnable --task regression \
            --model transformer --plot_loss --plot_accuracy

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

        # # Climax-Smooth: chooosing model by test loss. Should give same result as second command.
        # python main.py --comment "ClimaX on ${DATA}" \
        #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST_ERPEAFTER \
        #     --records_file REPROTEST_CLIMAX_${DATA}_TEST_ERPEAFTER.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_pattern TEST --epochs 1000 --lr 0.001 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
        #     --optimizer RAdam --pos_encoding learnable --relative_pos_encoding erpe --where_to_add_relpos after --task regression \
        #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
        #     --plot_loss --plot_accuracy

        # # Same as above but with --normalize_label
        # python main.py --comment "ClimaX on ${DATA}" \
        #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST_NORMALIZELABEL \
        #     --records_file REPROTEST_CLIMAX_${DATA}_TEST.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_pattern TEST --epochs 500 --lr 0.001 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
        #     --optimizer RAdam --pos_encoding learnable --task regression \
        #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
        #     --plot_loss --plot_accuracy --normalize_label

        # # Try SeqPool
        # python main.py --comment "ClimaX on ${DATA} SEQPOOL" \
        #     --seed $SEED --name REPROTEST_CLIMAX_${DATA}_TEST_SEQPOOL_LOCALMASK \
        #     --records_file REPROTEST_CLIMAX_${DATA}_TEST_SEQPOOL_LOCALMASK.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_pattern TEST --epochs 500 --lr 1e-2 \
        #     --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 256 \
        #     --optimizer RAdam --pos_encoding learnable_sin_init --task regression \
        #     --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
        #     --plot_loss --plot_accuracy --normalize_label --pool seqpool_multihead --local_mask 3
    done
done


# See https://arxiv.org/pdf/2010.02803 (Table 15)
# --val_pattern TEST means to use the TEST file as the validation set.
# --val_ratio 0.2  means to split another val set from the training set.