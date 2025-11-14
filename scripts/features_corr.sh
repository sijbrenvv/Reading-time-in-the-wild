#!/bin/bash

# Retrieve T-Scan data and add more features
python3 code/get_features_TScan.py -inp data/dvhn_exp_data_pre.json -out data/dvhn_exp_data_feat_ts.csv -hb "p312461" -pn "DvhN_rt_2024_test_20250804"

# Parse the input (add CoNLLU parses)
python3 code/parse.py -inp data/dvhn_exp_data_feat_ts.csv -out data/dvhn_exp_data_feat_ts_conllu.csv

# Run ProfilingUD on the CoNLLU files and add the features to the data
python3 code/get_features_profud.py -inp data/dvhn_exp_data_feat_ts_conllu.csv -out data/dvhn_exp_data_feat_profud.csv -hb "p312461"

# Correlation values
## T-Scan
# Compute multicollinearity
#python3 code/multicollinearity.py -inp data/dvhn_exp_data_feat_ts.csv -out_coll data/multicoll_ts.csv -out_vif data/vif_ts.csv
# Compute the correlation between the average reading time and other features
#python3 code/correlation.py -cf "avg_rt_view" -inp data/dvhn_exp_data_feat.csv -out data/dvhn_exp_data_corr_avgrtview.csv
python3 code/correlation.py -cf "avg_rt_tok" -inp data/dvhn_exp_data_feat_ts.csv -out data/dvhn_exp_data_ts_corr_avgrttok.csv

## Profiling-UD
#python3 code/multicollinearity.py -inp data/dvhn_exp_data_feat_profud.csv -out_coll data/multicoll_profud.csv -out_vif data/vif_profud.csv
python3 code/correlation.py -cf "avg_rt_tok" -inp data/dvhn_exp_data_feat_profud.csv -out data/dvhn_exp_data_profud_corr_avgrttok.csv
