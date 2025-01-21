# Local Mask + Multihead SeqPool
# Run `mkdir repro_output` first.
for DATA in AppliancesEnergy
do
    for SEED in 0 1 2
    do

        # Climax-Smooth: chooosing model by test loss. Should give same result as second command.
        python main.py --comment "ClimaX on ${DATA}" \
            --seed $SEED --name REPROTEST_CLIMAX_LOCALMASK1_${DATA}_TEST \
            --records_file output_repro/REPROTEST_CLIMAX_LOCALMASK1_${DATA}_TEST.xls \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --pattern TRAIN --val_pattern TEST --epochs 2000 --lr 0.001 --batch_size 128 \
            --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
            --optimizer RAdam --pos_encoding learnable --task regression \
            --model climax_smooth --patch_length 1 --stride 1 --smooth_attention \
            --plot_loss --plot_accuracy --local_mask 1 --pool seqpool_multihead
            # --relative_pos_encoding erpe --where_to_add_relpos after
    done
done
