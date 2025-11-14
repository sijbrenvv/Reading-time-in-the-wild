import argparse
import logging
import os
import time
import pandas as pd
import shutil
from pathlib import Path
from datasets import Dataset, concatenate_datasets
from minicons.scorer import IncrementalLMScorer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
from nltk import Text
from nltk.probability import FreqDist
import numpy as np
import stanza
import spacy_stanza
import spacy_udpipe
import fasttext.util
import textstat
import re
#from surprisal import AutoHuggingFaceModel
from tqdm import tqdm
from minicons import scorer
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from feat_utils import get_data, comp_read, load_surp_model

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (API keys and tokens)
load_dotenv()
# Load environment variable for HuggingFace
hf_token = os.getenv("HF_TOKEN")


def create_arg_parser() -> argparse.Namespace:
    """
    Creates an argument parser to handle command-line arguments.

    :return: argparse.Namespace: The parser arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        "-inp",
        required=True,
        help="Path to the input file.",
        type=str,
    )
    parser.add_argument(
        "--output_file_path",
        "-out",
        required=True,
        help="Path to the output file. Please specify .csv or .json.",
        type=str,
    )
    parser.add_argument(
        "--random_seed",
        "-seed",
        #required=True,
        help="The random seed to use.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--habrok-num",
        "-hb",
        default="",
        help="If true, tries to use /scratch/NUM to move and load models from.",
        type=str,
    )
    parser.add_argument(
        "--project_name",
        "-pn",
        default="DvhN_rt_2024_test_20250804",
        help="The T-Scan project name. Default = 'DvhN_rt_2024_test_20250804'",
        type=str,
    )
    return parser.parse_args()


def get_tscan_feat(dir: str, n_rows: int) -> Dataset:
    """ """

    df = None
    master_columns = None
    missing_files_count = 0
    missing_columns_count = 0

    for i in tqdm(range(n_rows)):
        file_path = f"{dir}{i}.txt.document.csv"

        try:
            file_df = pd.read_csv(file_path, index_col=False, na_values=["NA", ""], keep_default_na=True)
            file_df.columns = file_df.columns.str.strip()

            for col in file_df.columns:
                if col != "Inputfile":
                    file_df[col] = pd.to_numeric(file_df[col], errors="coerce")
            file_df = file_df.fillna(0.0)

            # If this is the first valid file, establish master columns
            if master_columns is None:
                master_columns = list(file_df.columns)
                df = pd.DataFrame(columns=master_columns)
                logger.info(f"Using columns from first valid file: {file_path}")

        except FileNotFoundError:
            missing_files_count += 1
            file_df = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            file_df = pd.DataFrame()

        # If columns were identified, standardize file_df to match
        if master_columns is not None:
            if file_df.empty:
                # create a zero row if the file is missing or empty
                file_df = pd.DataFrame([{col: (0.0 if col != "Inputfile" else "missing.txt") for col in master_columns}])
            else:
                missing_cols_in_file = set(master_columns) - set(file_df.columns)
                if missing_cols_in_file:
                    #logger.info(f"File {i}.txt.document.csv")
                    missing_columns_count += 1

                    # Prepare missing values respecting original dtype
                    missing_data = {}
                    for col in missing_cols_in_file:
                        if col == "Inputfile":
                            missing_data[col] = "missing.txt"
                        #if pd.api.types.is_numeric_dtype(column_dtypes[col]):
                        #    missing_data[col] = 0.0
                        # All other columns are floats or integers
                        else:
                            missing_data[col] = 0.0

                    # Add all missing columns at once
                    if missing_data:
                        missing_df = pd.DataFrame([missing_data] * len(file_df))
                        file_df = pd.concat([file_df, missing_df], axis=1)

                file_df = file_df[master_columns]

            # Take only the first row to keep alignment 1 file = 1 row
            file_df = file_df.iloc[[0]]
            df = pd.concat([df, file_df], ignore_index=True)
        else:
            continue

    if master_columns is None:
        raise RuntimeError(f"No valid files found in directory: {dir}")

    # Log summary of issues
    logger.info(
        f"T-Scan feature loading completed. Missing files: {missing_files_count}, "
        f"files with missing columns: {missing_columns_count}"
    )

    # Convert object columns to string explicitly
    #for col in df.select_dtypes(include="object").columns:
    #    df[col] = df[col].astype(str)

    return Dataset.from_pandas(df)

    """
    # Select T-Scan's top fifteen (complexity) features + inputfile and LiNT scores
    columns = ["wrd_freq_log_zn_corr", "wrd_freq_zn_log", "Conc_nw_ruim_p", "Conc_nw_strikt_p", "Alg_nw_d",
               "Pers_ref_d", "Pers_vnw_d", "Wrd_per_zin", "Wrd_per_dz", "Inhwrd_dz_zonder_abw", "AL_max",
               "Bijzin_per_zin", "Bijv_bep_dz_zbijzin", "Extra_KConj_dz", "MTLD_inhwrd_zonder_abw",
               "LiNT_score1", "LiNT_score2", "Word_per_doc", "Morf_per_wrd"]
    df = pd.DataFrame(columns=columns)

    for i in range(n_rows):
        #for c, file in enumerate(files):
            #if c == 1:
            #    break
            #logger.info(f"File: {c}; {file}")
        try:
            file_df = pd.read_csv(f"{dir}{i}.txt.document.csv", index_col=False)
            file_df.columns = file_df.columns.str.strip()

            # Fill missing columns with 0
            for col in columns:
                if col not in file_df.columns:
                    file_df[col] = 0.0
        except FileNotFoundError:
            file_df = pd.DataFrame([{col: 0.0 for col in columns}])
            #df = pd.concat([df, file_df], ignore_index=True)

        # Take only the first row to match 1 row per file
        if not file_df.empty:
            file_df = file_df.iloc[[0]]
        else:
            # create a zero row if the file is empty
            file_df = pd.DataFrame([{col: 0.0 for col in columns}])

        file_df = file_df[columns]
        df = pd.concat([df, file_df], ignore_index=True)

    return Dataset.from_pandas(df)"""


if __name__ == "__main__":

    args = create_arg_parser()

    # Check if the provided output path has a valid extension
    if os.path.splitext(args.output_file_path)[1] not in {".json", ".jsonl", ".csv"}:
        logger.info(f"Extension found: {os.path.splitext(args.output_file_path)[1]}")
        raise Exception(f"'{args.output_file_path}' contains no valid extension. Please use .json, .jsonl, or .csv.")

    # Set seed for replication
    set_seed(args.random_seed)

    # Get the data
    logger.info(f"Loading the data...")
    dataset = get_data(args.input_file_path)
    # Use first instance for testing
    #dataset = dataset.select(range(200))

    #logger.info(f"Sents first instance: \n {dataset['body'][0].splitlines()}")

    # Remove T-Scan skip markers
    dataset = dataset.map(
        lambda example:
        {"body": re.sub(r'### ', r'', example["body"])}
    )

    # Examine the dataset
    logger.info(f"Dataset shape: {dataset.shape}")
    #logger.info(f"Dataset: {dataset}")
    #logger.info(f"Dataset id: {dataset['id'][0:10]}")
    #logger.info(f"Dataset: {dataset[:2]}")
    #exit()

    # Retrieve T-Scan analyses
    temp_ds = get_tscan_feat(f"exp/TScan_output/{args.project_name}/", dataset.shape[0])
    logger.info(f"TScan shape: {temp_ds.shape}")

    # Merge datasets
    tsfeat_ds = concatenate_datasets([dataset, temp_ds], axis=1)
    logger.info(f"tsfeat_ds shape: {tsfeat_ds.shape}")
    #tsfeat_ds = temp_ds
    del temp_ds

    tsfeat_ds = tsfeat_ds.filter(lambda example: sum(example[col] for col in ["Word_per_doc", "Morf_per_wrd"]) != 0)
    logger.info(f"Shape after removing T-Scan misparses: {tsfeat_ds.shape}")


    logger.info("Adding readability scores...")
    feat_dataset = tsfeat_ds.map(
        comp_read
        #remove_columns=["syll_count", "avg_syll_tok"]
        #num_proc=1  # Use single-threaded processing (FastText model cannot be serialised)
    )

    logger.info(f"Feat dataset shape: {feat_dataset.shape}\n Adding surprisal scores...")
    #logger.info(f"Feat dataset columns: {feat_dataset.column_names}")


    # Add average reading time
    # PyArrow cannot deal with mixed data types, and expects bytes (string)
    feat_dataset = feat_dataset.map(
        lambda example: {
            "avg_rt_view": str(float(example["read_time_sec"]) / int(example["views"])),
            #"avg_rt_tok": str((float(example["read_time_sec"]) / int(example["views"])) / int(example["Word_per_doc"]))
            "avg_rt_tok": str((float(example["read_time_sec"]) / int(example["views"])) / int(len(example["body"].split(" "))))
        },
        remove_columns=["id"]
    )

    # logger.info("Postprocessing...")
    # Remove articles where: 1. Word_per_doc < 100, and 2. avg_rt_tok < .15 (avg. rt. native English--> .327) \
    #feat_dataset = feat_dataset.filter(lambda example: int(example["Word_per_doc"]) >= 100)
    feat_dataset = feat_dataset.filter(lambda example: float(example["avg_rt_tok"]) >= .15)

    # Surprisal
    models = ["BramVanroy/fietje-2", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B", "utter-project/EuroLLM-9B",
              "ai-forever/mGPT", "Tweeties/tweety-7b-dutch-v24a"]
    for model_name in models:
        ilm_model = load_surp_model(model_name, hf_token)
        logger.info(f"Model: {model_name.split('/')[1]}")
        feat_dataset = feat_dataset.map(
            lambda example: {
                f"surp_{model_name.split('/')[1]}": np.mean(ilm_model.sequence_score(example["body"].splitlines(), reduction=lambda x: -x.sum(0).item()))
            }
        )

    logger.info("Postprocessing...")
    # Add baseline reading time
    #mean_avgrttok = np.mean(list(map(float, tsfeat_ds["avg_rt_tok"])))
    #logger.info(f"Mean avgrttok: {mean_avgrttok}")
    #feat_dataset = tsfeat_ds.map(lambda example: {"baseline_rt": str(mean_avgrttok * int(example["Word_per_doc"]))})

    rem_feat = [
        feat for feat in tqdm(feat_dataset.features.keys())
        if len(set(feat_dataset[feat])) == 1  # Check if only 1 unique value
        # Check if default values are over 10% of occurrences
        or (feat_dataset[feat].count("0.00000") / len(feat_dataset[feat])) > .10
        or (feat_dataset[feat].count("NA") / len(feat_dataset[feat])) > .10
    ]
    feat_dataset = feat_dataset.remove_columns(rem_feat)
    logger.info(f"Removed features due to constant values or excessive default values (>10%): {rem_feat}")
    feat_df = feat_dataset.to_pandas()

    # Remove outliers on average reading time per token
    read_tok = feat_df["avg_rt_tok"]
    out_threshold = np.mean(read_tok) + 3 * np.std(read_tok)
    out_ind = np.where(np.abs(read_tok) > out_threshold)[0]
    logger.info(f"Outlier Threshold Used: {out_threshold}")
    logger.info(f"Number of Outliers Found: {len(out_ind)}")
    feat_df = feat_df.drop(out_ind, axis=0)

    feat_dataset = Dataset.from_pandas(feat_df)

    logger.info(f"Data shape after removal: {feat_dataset.shape}")
    #logger.info(f"Feat dataset columns: {feat_dataset.column_names}")
    #exit()

    # Export dataset
    logger.info(f"Exporting the processed dataset...")
    # os.makedirs(args.output_file_path, exist_ok=True)
    if args.output_file_path.endswith(".csv"):
        feat_dataset.to_csv(args.output_file_path, index=False, sep=",")
        logger.info(f"Dataset saved to: {args.output_file_path}")
    if args.output_file_path.endswith(".json"):
        feat_dataset.to_json(args.output_file_path, orient="records", lines=True)
        logger.info(f"Dataset saved to: {args.output_file_path}")

