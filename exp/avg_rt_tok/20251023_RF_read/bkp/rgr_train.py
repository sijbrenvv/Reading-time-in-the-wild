import argparse
import joblib
import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from transformers import set_seed
import logging
import os
import warnings
from rgr_utils import get_data

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
        help="Model to use. See choices for the choices. Default: 'RF'",
        choices=["linSVR", "SVR", "polySVR", "HGBT", "RF"],
        default="RF",
    )
    parser.add_argument(
        "-tr",
        "--train_file",
        type=str,
        help="File containing the training data.",
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Path where to save the trained model.",
    )
    parser.add_argument(
        "-dp",
        "--dependent_variable",
        type=str,
        default="avg_rt_view",
        help="The dependent/response variable. Default='avg_rt_view'.",
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
    """
    Main function to train and save the classifier model.
    """
    args = create_arg_parser()

    # Set seed for replication
    set_seed(args.random_seed)

    # Get the train data
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Train file '{args.train_file}' not found.")
    logger.info("Loading the train data...")
    x_train, y_train = get_data(args.train_file, args.dependent_variable)

    x_train = x_train.to_numpy()

    regressors = {
        #"LR": LinearRegression,
        "linSVR": SVR(kernel="linear"),
        "polySVR": SVR(kernel="poly"),
        "SVR": SVR(kernel="rbf"),
        "HGBT": HistGradientBoostingRegressor(random_state=args.random_seed),
        "RF": RandomForestRegressor(n_jobs=-1, random_state=args.random_seed),
    }

    # Select the given regressor
    logger.info("Loading model...")
    model = regressors[args.model]

    # Train the model
    logger.info("Training model...")
    model.fit(x_train, y_train)

    # Save the model
    logger.info("Saving model...")
    output_path = os.path.join(args.output_file, "model.pkl")
    os.makedirs(args.output_file, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
