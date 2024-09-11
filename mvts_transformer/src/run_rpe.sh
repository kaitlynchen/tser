# Usage: ./run_rpe.sh AppliancesEnergy

DATA=${1}
for POSENC in none
do
    for LAM in 0
    do
        for SEED in 0 1 2
        do
            python main.py --comment "ClimaX ${DATA}" \
                --seed $SEED --name ClimaX_smooth_${DATA}_BASELINE \
                --records_file ${DATA}_20240810_BASELINE.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr 0.001 \
                --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
                --optimizer RAdam --pos_encoding learnable --relative_pos_encoding $POSENC \
                --task regression --model climax_smooth --patch_length 1 --stride 1 --reg_lambda $LAM --smooth_attention \
                --lambda_posenc_smoothness 0 --plot_loss --plot_accuracy
        done
    done
done


# --where_to_add_relpos after_gating



# Note: --pattern TRAIN --val_ratio 0.2 validates on 20% of train set
# Note: --pattern TRAIN --val_pattern TEST validates on test set
