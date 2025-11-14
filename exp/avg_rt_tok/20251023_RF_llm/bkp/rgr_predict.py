import argparse
from transformers import set_seed
import joblib
import logging
import os
import warnings
from rgr_utils import get_data
import pandas as pd

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
        "-m",
        "--model",
        type=str,
        help="Path to the trained model.",
    )
    parser.add_argument(
        "-inp",
        "--input_file",
        type=str,
        help="Input file containing the data.",
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Path where to save the model predictions.",
    )
    parser.add_argument(
        "-dp",
        "--dependent_variable",
        type=str,
        default="avg_rt_norm",
        help="The dependent/response variable. Default='avg_rt_norm'.",
    )
    parser.add_argument(
        "--random_seed",
        "-seed",
        help="The random seed to use.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--block_of_feature",
        "-bof",
        help="Select which block of features to analyse, including the dataset. Default: 'tscan'.",
        type=str,
        choices=["prof-ud", "tscan", "read", "llm", "metadata"],
        default="tscan"
    )

    return parser.parse_args()


def main() -> None:

    args = create_arg_parser()

    # Set seed for replication
    set_seed(args.random_seed)

    # Get the dev data
    logger.info("Loading input data...")
    x_dev, _ = get_data(args.input_file, args.dependent_variable)

    x_dev = x_dev.to_numpy()

    # Load the trained model
    logger.info("Loading model...")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file '{args.model}' not found.")
    model = joblib.load(args.model)

    # Use the loaded model to predict on dev data
    logger.info("Making predictions...")
    y_pred = model.predict(x_dev)

    # Save predictions to predefined output file
    logger.info("Saving predictions to output file...")
    with open(args.output_file, "w") as file:
        file.write("\n".join(map(str, y_pred)))


if __name__ == "__main__":
    main()
