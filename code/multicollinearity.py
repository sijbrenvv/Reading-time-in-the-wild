import argparse
import logging
import os
from datasets import Dataset
from transformers import set_seed
import warnings
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "--output_file_path_coll",
        "-out_coll",
        required=True,
        help="Path to the output file for the multicollinearity scores. Please specify .csv or .json.",
        type=str,
    )
    parser.add_argument(
        "--output_file_path_vif",
        "-out_vif",
        required=True,
        help="Path to the output file for the VIF scores. Please specify .csv or .json.",
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

    return parser.parse_args()


def get_data(inp_file: str) -> Dataset:
    """
    Load the data from a CSV or JSON(L) file into a HF Dataset object

    :param inp_file: The path the to the data file with a CSV or JSON(L) extension.
    :return: The data as a HF Dataset object.
    """

    if not os.path.exists(inp_file):
        raise FileNotFoundError(f"CSV or JSON input file '{inp_file}' not found")

    # Check if the provided input path has a valid extension
    if os.path.splitext(inp_file)[1] not in {".json", ".jsonl", ".csv"}:
        raise Exception(f"'{inp_file}' contains no valid extension. Please provide a JSON(L) or CSV file.")

    if inp_file.endswith(".csv"):
        return Dataset.from_csv(inp_file)
    elif inp_file.endswith(".json") or inp_file.endswith(".jsonl"):
        return Dataset.from_json(inp_file)


def all_corr(feat_ds: Dataset) -> pd.DataFrame:
    """
    Computes the correlation between all features and outputs them in descending order.
    Allows for easy multicollinearity check.

    :param feat_ds: The Dataset object with the features and values.
    :return: Pandas Dataframe: An ordered fd with all non-self correlation values.
    """

    # Compute correlation matrix, convert to long format, name columns, remove self-correlations and duplicates, \
    # sort by absolute correlation in descending order, and reset index (add minus).
    df = feat_ds.to_pandas()
    corr_matrix = df.corr()
    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ["Feature 1", "Feature 2", "Pearson r"]
    corr_pairs = corr_pairs[corr_pairs["Feature 1"] != corr_pairs["Feature 2"]]
    # Sort the feature pairs to ensure uniqueness (avoid Feature 1, Feature 2 and Feature 2, Feature 1 being considered different)
    corr_pairs["sorted_tuple"] = corr_pairs.apply(lambda x: tuple(sorted([x["Feature 1"], x["Feature 2"]])), axis=1)
    corr_pairs = corr_pairs.drop_duplicates(subset="sorted_tuple").drop(columns=["sorted_tuple"])
    corr_pairs = corr_pairs.sort_values(by="Pearson r", ascending=False, key=abs)
    corr_pairs = corr_pairs.reset_index(drop=True)

    return corr_pairs


def compute_vif(feat_ds: Dataset, features: list) -> pd.DataFrame:
    """"""
    # ref 1: https://www.statology.org/multicollinearity-in-python/
    # ref 2: https://towardsdatascience.com/targeting-multicollinearity-with-python-3bd3b4088d0b
    # ref 3: https://github.com/Darwinkel/shared-task-semeval2024/blob/main/feature_analysis/feature_multicollinearity.ipynb

    df = feat_ds.to_pandas()
    X = df[features].select_dtypes(include=[np.number])
    #X = df[features]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    X = X.copy()
    # The calculation of variance inflation requires a constant
    X["intercept"] = 1

    # Create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif["Variable"] != "intercept"]
    vif = vif.sort_values("VIF", ascending=False)
    return vif

if __name__ == "__main__":

    args = create_arg_parser()

    # Check if the provided output path has a valid extension
    if os.path.splitext(args.output_file_path_coll)[1] not in {".json", ".jsonl", ".csv"}:
        logger.info(f"Extension found: {os.path.splitext(args.output_file_path_coll)[1]}")
        raise Exception(f"'{args.output_file_path_coll}' contains no valid extension. Please use .json, .jsonl, or .csv.")

    # Check if the provided output path has a valid extension
    if os.path.splitext(args.output_file_path_vif)[1] not in {".json", ".jsonl", ".csv"}:
        logger.info(f"Extension found: {os.path.splitext(args.output_file_path_vif)[1]}")
        raise Exception(f"'{args.output_file_path_vif}' contains no valid extension. Please use .json, .jsonl, or .csv.")

    # Set seed for replication
    set_seed(args.random_seed)

    # Get the data
    logger.info(f"Loading the data...")
    dataset = get_data(args.input_file_path)

    # Remove irrelevant non-predictive features (metadata) and readability features
    dataset = dataset.remove_columns(
        ["title",
         "publication_date",
         "body", "views",
         "read_time_sec",
         "avg_rt_tok",
         "avg_rt_view",
         "Inputfile"
         #"word_count_category",
         #"brouwer_index",
         #"flesch_douma",
         #"mcalpine",
         #"flesch_reading",
         #"day_of_week"
         ]
    )

    # Compute correlations and VIF scores for all features
    corr_scores = all_corr(feat_ds=dataset)
    vif_scores = compute_vif(feat_ds=dataset, features=dataset.column_names)

    # Export dataset
    logger.info(f"Exporting correlation and VIF datasets...")
    if args.output_file_path_coll.endswith(".csv"):
        corr_scores.to_csv(args.output_file_path_coll, index=False, sep=",")
        vif_scores.to_csv(args.output_file_path_vif, index=False, sep=",")
        logger.info(f"Datasets saved to: {args.output_file_path_coll} and {args.output_file_path_vif}")
    if args.output_file_path_coll.endswith(".json"):
        corr_scores.to_json(args.output_file_path_coll, orient="records", lines=True)
        vif_scores.to_json(args.output_file_path_vif, orient="records", lines=True)
        logger.info(f"Datasets saved to: {args.output_file_path_coll} and {args.output_file_path_vif}")