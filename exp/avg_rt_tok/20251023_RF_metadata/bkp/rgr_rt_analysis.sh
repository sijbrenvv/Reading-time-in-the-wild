#!/bin/bash

# Experiment folder is partially a command line argument
# First argument is the dependent variable to use
# Second argument is a random directory name to save the experiment
exp_fol="exp/$1/$2"

# Check if folder already exists, if so, exit, if not, continue
if [ -d "$exp_fol" ]; then
    echo "Experiment folder already exists. Exiting..."
    exit 1
fi

# Set variables
log=${exp_fol}/log/
out=${exp_fol}/out/
mod=${exp_fol}/mod/
bkp=${exp_fol}/bkp/

# Create folders in exp folder
mkdir -p $exp_fol $log $out $mod $bkp

# Define experiment data
#exp_data="habrok_output/dvhn_exp_data_feat.csv"
exp_data="data/dvhn_exp_data_feat_ts.csv"

# Set dependent variable
dep_var=$1
bof="metadata"

# Call rgr_create_splits.py to create the data splits
python3 code/rgr/rgr_create_splits.py -d $exp_data -out "${out}" -dp $dep_var -dow "all" -bof $bof # -ef flesch_douma brouwer_index
train="${out}train.json"
test="${out}test.json"

# Train, predict, and evaluate
python3 code/rgr/rgr_train.py -tr $train -m "RF" -out "${mod}" -dp $dep_var -bof $bof
python3 code/rgr/rgr_predict.py -inp $test -m "${mod}model.pkl" -out "${out}pred.txt" -dp $dep_var -bof $bof
python3 code/rgr/rgr_evaluate.py -t $test -p "${out}pred.txt" -m "${mod}model.pkl" -dp $dep_var -bof $bof -fig "${out}" > ${log}eval.txt

# Backup scripts
cp scripts/rgr_rt_analysis.sh code/rgr/rgr_train.py code/rgr/rgr_predict.py code/rgr/rgr_evaluate.py code/rgr/rgr_create_splits.py code/rgr/rgr_utils.py $bkp