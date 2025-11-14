#!/bin/bash

# Preprocess data for T-Scan
python3 code/preprocess_TScan.py -inp data/dvhn_exp_data.json -out data/dvhn_exp_data_pre.json

# Stage data for T-Scan and start scan
#python3 code/stager_TScan.py -inp data/dvhn_exp_data_pre.json -pn "DvhN_rt_2024_test_20250804"
