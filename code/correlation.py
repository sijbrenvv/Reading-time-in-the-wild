import argparse
import logging
import os
from datasets import Dataset
from transformers import set_seed
import warnings
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

le = LabelEncoder()

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
        "--correlation_feature",
        "-cf",
        #required=True,
        help="The feature to compute the correlations to. Default = 'avg_rt_view'.",
        default="avg_rt_view",
        type=str,
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


def comp_corr(feat_ds, cols, corr_feat) -> Dataset:  # dict[str:tuple[any:any]]
    """
    Computes Pearson's r (correlation) with p-value between the average reading time and other features.

    :param feat_ds: The Dataset object with the features and values.
    :param cols: The columns to include for the correlation calculation.
    :param corr_feat: The feature to compute the correlation with.
    :return: Dataset object: A new Dataset object with the calculated Pearson's r and p-value.
    """

    # Using a dict comprehension, iterate over the features, compute the correlation and p-value \
    # Convert the dict into a HF Dataset object
    #return Dataset.from_dict({feat: pearsonr(feat_ds[corr_feat], feat_ds[feat]) for feat in cols})

    df = feat_ds.to_pandas()

    # Convert the target column to numeric
    y = pd.to_numeric(df[corr_feat], errors='coerce').replace([np.inf, -np.inf], np.nan)

    results = []
    for feat in cols:
        if feat == corr_feat:
            continue
        # Convert to numeric
        x = pd.to_numeric(df[feat], errors='coerce').replace([np.inf, -np.inf], np.nan)
        valid = x.notna() & y.notna()

        if valid.sum() < 2:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(x[valid], y[valid])

        results.append({"Feature": feat, "Pearson_r": r, "p_value": p})

    return Dataset.from_pandas(pd.DataFrame(results))



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

    # Check if provided feature exists in the data
    corr_feat = args.correlation_feature
    try:
        # Determine the feature columns
        avg_rt_idx = dataset.column_names.index(corr_feat)
    except ValueError:
        raise ValueError(f"'{args.correlation_feature}' is not present in the data. Please check for typos or consult the data.")

    # Remove unnecessary metadata and readability features
    dataset = dataset.remove_columns(["title", "publication_date", "body"]) # , "id", "__index_level_0__"

    # Compute correlation and corresponding p-values
    logger.info(f"Computing the correlation between {corr_feat} and other features...")
    num_cols = [
        c for c in dataset.column_names
        if pd.api.types.is_numeric_dtype(dataset.to_pandas()[c])
    ]
    corr_dataset = comp_corr(feat_ds=dataset, cols=num_cols, corr_feat=corr_feat)

    # Sort based on significance
    #feature_names = list(corr_dataset.features.keys())
    #p_values = [corr_dataset[feature][1] for feature in feature_names]
    # Sort features by p-value
    #sorted_features = sorted(feature_names, key=lambda feat: corr_dataset[feat][1])
    #sorted_dataset = Dataset.from_dict({feat: corr_dataset[feat] for feat in sorted_features})

    corr_df = corr_dataset.to_pandas()
    corr_df["p_value"] = pd.to_numeric(corr_df["p_value"], errors="coerce")
    sorted_corr_df = corr_df.sort_values("p_value", ascending=True)
    sorted_dataset = Dataset.from_pandas(sorted_corr_df)

    # Export dataset
    logger.info(f"Exporting the correlation dataset...")
    if args.output_file_path.endswith(".csv"):
        sorted_dataset.to_csv(args.output_file_path, index=False, sep=",")
        logger.info(f"Dataset saved to: {args.output_file_path}")
    if args.output_file_path.endswith(".json"):
        sorted_dataset.to_json(args.output_file_path, orient="records", lines=True)
        logger.info(f"Dataset saved to: {args.output_file_path}")
