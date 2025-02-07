# Usage: ./run.sh AppliancesEnergy

DATA=${1}

for SEED in 0 1 2
do
    for REG in 0 1e-5 1e-4 1e-3 1e-2 1e-1 0
    do

        # Baseline
        python main.py --comment "ClimaX ${DATA}" \
            --seed $SEED --name ClimaX_smooth_${DATA}_L1REG \
            --records_file ${DATA}_20240810_L1REG.xls \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr 0.001 \
            --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
            --optimizer RAdam --pos_encoding learnable --relative_pos_encoding none \
            --task regression --model climax_smooth --patch_length 1 --stride 1 --reg_lambda 0 --smooth_attention \
            --lambda_posenc_smoothness 0 --plot_loss --plot_accuracy --l1_reg $REG

        # # CONVIT + Attnsmooth
        # python main.py --comment "ClimaX ${DATA}" \
        #     --seed $SEED --name ClimaX_smooth_${DATA}_CONVIT_ATTNSMOOTH \
        #     --records_file ${DATA}_20240810_CONVIT_ATTNSMOOTH.xls \
        #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
        #     --pattern TRAIN --val_ratio 0.2 --epochs 500 --lr 0.001 \
        #     --num_layers 3 --num_heads 16 --d_model 128 --dim_feedforward 256 \
        #     --optimizer RAdam --pos_encoding learnable --relative_pos_encoding custom_rpe --where_to_add_relpos after_gating \
        #     --task regression --model climax_smooth --patch_length 1 --stride 1 --reg_lambda 0.01 --smooth_attention \
        #     --lambda_posenc_smoothness 0 --plot_loss --plot_accuracy
    done
done




# Note: --pattern TRAIN --val_ratio 0.2 validates on 20% of train set
# Note: --pattern TRAIN --val_pattern TEST validates on test set
