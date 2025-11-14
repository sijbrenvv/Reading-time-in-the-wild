import argparse
import logging
import os
from transformers import set_seed
import warnings
from ProfilingUD_Code.ling_monitoring import use_in_pipe
from feat_utils import get_data, comp_read
from datasets import Dataset

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable needed to ensure equal feature dimensionality
observed_features = set()


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

    return parser.parse_args()


def comp_total_ud_features(example: dict[str, any], idx: int):
    """
    Computes the total number of features returned by the UD Profiling pipeline \
    for the given dataset (inconsistent features per entry), \
    and stores these features in a global variable.

    :param example: Each row (data point/sample) of the Dataset object.
    :param idx: Index of the processed file.
    :return: Keeps track of every observed feature.
    """

    global observed_features
    ud_vect = use_in_pipe(example["conllu"], str(idx + 1))
    featstring, _, _ = ud_vect.split("\n")
    feats = featstring.split("\t")
    observed_features.update(feats)


def ud_features(example: dict[str, any], idx: int) -> dict[str:str]:
    """
    Retrieves the features computed by Profiling-UD, and processes them into the Dataset object.

    :param example: Each row (data point/sample) of the Dataset object.
    :param idx: Index of the processed file.
    :return: example: dictionary with the added features and corresponding values.
    """

    global observed_features
    ud_vect = use_in_pipe(example["conllu"], str(idx + 1))
    featstring, valuestring, _ = ud_vect.split("\n")
    feats = featstring.split("\t")
    values = valuestring.split("\t")

    # Add the features directly to the dataset (example), and ignore the identifier
    for c, feat in enumerate(feats):
        if feat != "identifier":
            example[feat] = values[c]

    # Fill in missing features with a default value of 0.0
    for obs_feat in observed_features:
        if obs_feat not in example:
            example[obs_feat] = str(0.0)

    return example


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

    # Check None values
    #test_df = dataset.to_pandas()
    #print(test_df.isna().sum())  # Counts missing values per column

    # Remove the T-Scan features
    temp_df = dataset.to_pandas()
    temp_df = temp_df[["publication_date", "day_of_week", "domain", "title", "body", "sentiment_label_detailed", "Inputfile",
                       "classification", "views", "read_time_sec", "avg_rt_view", "avg_rt_tok", "brouwer_index",
                       "flesch_douma", "LiNT_score1", "LiNT_score2", "surp_fietje-2", "surp_EuroLLM-9B", "surp_mGPT",
                       "surp_tweety-7b-dutch-v24a", "syll_count", "avg_syll_tok", "surp_Qwen3-4B", "surp_Qwen3-1.7B", "conllu"]]
    dataset = Dataset.from_pandas(temp_df)
    del temp_df

    # Examine the dataset
    logger.info(f"Dataset shape: {dataset.shape}")

    dataset.map(
        comp_total_ud_features,
        with_indices=True
        #num_proc=1  # Use single-threaded processing
    )

    # Extract ProfilingUD features
    logger.info("Extracting ProfilingUD features from the articles...")
    udfeat_dataset = dataset.map(
        ud_features,
        with_indices=True,
        remove_columns=["conllu"]
        #num_proc=1  # Use single-threaded processing
    )

    # Add average reading time
    # PyArrow cannot deal with mixed data types, and expects bytes (string)
    #udfeat_dataset = udfeat_dataset.map(
    #    lambda example: {
    #        "avg_rt_view": str(float(example["read_time_sec"]) / int(example["views"])),
    #        "avg_rt_tok": str((float(example["read_time_sec"]) / int(example["views"])) / int(len(example["body"].split(" "))))
    #    },
    #    remove_columns=["id", "__index_level_0__"]
    #)

    # logger.info("Postprocessing...")
    # Remove articles where: 1. n_tokens < 100, and 2. avg_rt_tok < .15 (avg. rt. native English--> .327) \
    #udfeat_dataset = udfeat_dataset.filter(lambda example: int(example["n_tokens"]) >= 100)
    #udfeat_dataset = udfeat_dataset.filter(lambda example: float(example["avg_rt_tok"]) >= .15)

    # Extract more features
    #logger.info("Computing readability...")
    #feat_dataset = udfeat_dataset.map(
    #    comp_read,
    #    #num_proc=1,  # Use single-threaded processing (FastText model cannot be serialised)
    #    remove_columns=["syll_count", "avg_syll_tok"]
    #)

    logger.info(f"udfeat_dataset shape: {udfeat_dataset.shape}")

    logger.info("Postprocessing...")
    # Add baseline reading time
    #mean_avgrttok = np.mean(list(map(float, feat_dataset["avg_rt_tok"])))
    #logger.info(f"Mean avgrttok: {mean_avgrttok}")
    #feat_dataset = feat_dataset.map(lambda example: {"baseline_rt": str(mean_avgrttok * int(example["n_tokens"]))})

    rem_feat = [
        feat for feat in udfeat_dataset.features.keys()
        if len(set(udfeat_dataset[feat])) == 1  # Check if only 1 unique value
        or (udfeat_dataset[feat].count("0.0") / len(udfeat_dataset[feat])) > .10  # Check if default value over 25% of occurrences
    ]
    feat_dataset = udfeat_dataset.remove_columns(rem_feat)
    logger.info(f"Removed features due to constant values or excessive default values (>10%): {rem_feat}")
    logger.info(f"{feat_dataset.shape}")

    # Export dataset
    logger.info(f"Exporting the processed dataset...")
    # os.makedirs(args.output_file_path, exist_ok=True)
    if args.output_file_path.endswith(".csv"):
        feat_dataset.to_csv(args.output_file_path, index=False, sep=",")
        logger.info(f"Dataset saved to: {args.output_file_path}")
    if args.output_file_path.endswith(".json"):
        feat_dataset.to_json(args.output_file_path, orient="records", lines=True)
        logger.info(f"Dataset saved to: {args.output_file_path}")

