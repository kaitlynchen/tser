# Scikit-learn baselines. Usage:
# ./run_sklearn.sh AppliancesEnergy
# # Josh's results:


# IEEEPPG LiveFuelMoistureContent BeijingPM10Quality
DATA=${1}

# for MODEL in lasso ridge xgboost random_forest
for MODEL in random_forest 
do
    for SEED in 0 1 2
    do
        python main_sklearn.py --comment "SIMPLE_${MODEL}_${DATA}" \
            --seed $SEED --name "SIMPLE_${MODEL}_${DATA}" \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/TSER/$DATA/ --data_class tsra \
            --records_file output_repro/SIMPLE_${MODEL}_${DATA} \
            --pattern TRAIN --test_pattern TEST  \
            --task regression --model $MODEL

    done
done