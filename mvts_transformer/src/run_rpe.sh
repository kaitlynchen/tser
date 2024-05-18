# Call ./run_rpe.sh AppliancesEnergy

DATA=${1}
for POSENC in erpe_alibi_init
do
    for LAM in 0
    do
        for SEED in 0
        do
            python main.py --comment "ClimaX ${DATA}" \
                --seed $SEED --name ClimaX_smooth_${DATA}_ERPEALIBI_ATTNSMOOTH \
                --records_file ${DATA}_debug.xls \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
                --pattern TRAIN --val_pattern TEST --epochs 1000 --lr 0.001 \
                --num_layers 3 --num_heads 8 --d_model 128 --dim_feedforward 512 \
                --optimizer RAdam  --pos_encoding learnable --relative_pos_encoding $POSENC --task regression \
                --model climax_smooth --patch_length 1 --stride 1 --reg_lambda $LAM --smooth_attention \
                --lambda_posenc_smoothness 0 --plot_loss --plot_accuracy
        done
    done
done


# Note: --pattern TRAIN --val_ratio 0.2 validates on 20% of train set
# Note: --pattern TRAIN --val_pattern TEST validates on test set
