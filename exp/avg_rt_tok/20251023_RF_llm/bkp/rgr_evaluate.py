import argparse

from scipy.stats import pearsonr
from transformers import set_seed
import logging
import os
import warnings
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from rgr_utils import get_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

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
        "-t",
        "--test_data",
        type=str,
        help="File containing the test data.",
    )
    parser.add_argument(
        "-p",
        "--prediction_data",
        type=str,
        help="Path to the model predictions.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to the trained model.",
    )
    parser.add_argument(
        "-dp",
        "--dependent_variable",
        type=str,
        default="avg_rt_norm",
        help="The dependent/response variable. Default='avg_rt_norm'.",
    )
    parser.add_argument(
        "-seed",
        "--random_seed",
        help="The random seed to use.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-fig",
        "--figure_save",
        type=str,
        help="Path to store the evaluation plots.",
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

    # Load the trained model
    logger.info("Loading model...")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file '{args.model}' not found.")
    model = joblib.load(args.model)

    # Load test data and predictions
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data file '{args.test_data}' not found.")
    if not os.path.exists(args.prediction_data):
        raise FileNotFoundError(f"Prediction data file '{args.prediction_data}' not found.")

    logger.info("Loading test data and predictions...")
    try:
        x_test, y_test = get_data(args.test_data, args.dependent_variable)
        with open(args.prediction_data) as file:
            y_pred = [float(line.strip()) for line in file]
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    #logger.info(f"x_test cols: {x_test.columns}")
    #logger.info(f"x_test shape: {x_test.shape}")
    #logger.info(f"x_test first index: {x_test[0]}")
    #logger.info(f"x_test first index read cols: {x_test[0, [27,31]]}")

    #exit()

    #if args.block_of_feature == "prof-ud":
        # Extract readability metrics from x_test and delete them from x_test (we do not use them as predictors)
    #    x_read = x_test[["brouwer_index", "flesch_douma", "flesch_reading", "mcalpine"]]

    #logger.info(f"x_read cols: {x_read.columns}")
    #logger.info(f"x_read shape: {x_read.shape}")
    #logger.info(f"x_read len brouwer_index: {len(x_read['brouwer_index'])}")
    #logger.info(f"len y_pred: {len(y_pred)}")
    #if args.block_of_feature == "prof-ud":
    #    x_test = x_test.drop(
    #        columns=["brouwer_index", "flesch_douma", "flesch_reading", "mcalpine"]
    #    ).to_numpy()
    #else:
    #    x_test = x_test.to_numpy()

    x_test = x_test.to_numpy()

    # Convert y_test to numpy
    y_test = y_test.to_numpy()

    # Search for outliers
    # Compute residuals (errors)
    residuals = y_test - y_pred

    # Define threshold as 3 standard deviations above the mean residual
    residual_threshold = np.mean(residuals) + 3 * np.std(residuals)

    # Identify outliers
    outlier_indices = np.where(np.abs(residuals) > residual_threshold)[0]

    print(f"Average reading time per token (y_test): {np.mean(y_test)}")
    print(f"Predicted average reading time per token (y_pred): {np.mean(y_pred)}")
    print(f"Outlier Threshold Used: {residual_threshold}")
    print(f"Number of Outliers Found: {len(outlier_indices)}")
    # print(f"Outliers at: {outlier_indices}")

    #outl_res = residuals[outlier_indices]
    #outl_pred = y_pred[outlier_indices]
    #outl_true = y_test[outlier_indices]
    #outl_titles = x_test["title"][outlier_indices]
    #outl_views = x_test["views"][outlier_indices]

    #outl_df = pd.DataFrame({
    #    "title": outl_titles,
    #    "views": outl_views,
    #    "residual": outl_res,
    #    "predicted": outl_pred,
    #    "gold": outl_true
    #})
    # Export outliers dataframe
    #outl_df.to_csv(f"{args.figure_save}outliers_df.csv", index=False, sep=",")

    # Remove outlier indices from x_test, y_test, y_pred, and recompute residuals
    logger.info(f"Removing outliers from data...")
    x_test = np.delete(x_test, outlier_indices, axis=0)
    y_test = np.delete(y_test, outlier_indices)
    y_pred = np.delete(y_pred, outlier_indices)
    residuals = y_test - y_pred

    print(f"Predicted average reading time per token (y_pred) after outlier removal: {np.mean(y_pred)}")

    # Evaluate predictions
    logger.info("Evaluating predictions...")
    print(f"R^2: {model.score(x_test, y_test)}")
    print(f"MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred)}")
    print(f"MAE: {mean_absolute_error(y_true=y_test,y_pred=y_pred)}")
    print(f"Correlation gold -- pred: {pearsonr(y_test, y_pred)}")

    #if args.block_of_feature == "prof-ud":
    #    x_read = x_read.drop(index=outlier_indices)
    #    # Compute correlation predicted value with all readability metrics
    #    for col in x_read.columns:
    #        print(f"Correlation pred -- {col}: {pearsonr(y_pred, x_read[col])}")

    # Compute standard deviation & standard error
    std_dev = np.std(residuals, ddof=1)
    std_err = std_dev / np.sqrt(len(residuals))

    print(f"Standard Deviation Residuals: {std_dev}")
    print(f"Standard Error Residuals: {std_err}")

    # Add intercept and coefficients if present
    if hasattr(model, "intercept_"):
        print(f"Intercept: {model.intercept_}")
    if hasattr(model, "coef_"):
        print(f"Slope Coefficients: {model.coef_}")


    # Visualise residuals
    # Residual distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color="blue")
    plt.axvline(0, color='red', linestyle='dashed')  # Reference line at zero
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig(f"{args.figure_save}residuals_dist.png")

    # Residuals and predictions
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='dashed')  # Reference line at zero
    plt.title("Residuals vs. Predictions")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.savefig(f"{args.figure_save}res_pred.png")

    # Residual boxplot
    plt.figure(figsize=(5, 6))
    sns.boxplot(y=residuals)
    plt.title("Boxplot of Residuals")
    plt.ylabel("Residuals")
    plt.savefig(f"{args.figure_save}res_box.png")

    # Compute and visualise feature importance (MDI) and permutation importance
    # Ref 1: https://github.com/Darwinkel/shared-task-semeval2024/blob/main/subtaskAB/classicml_baseline.py
    # Ref 2: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html

    # Read feature names from output
    with open(f"{args.figure_save}features.txt") as file:
        # Exclude the dependent variable
        feat_names = [str(line.strip()) for line in file][:-1]
        if args.block_of_feature == "prof-ud":
            feat_names = [x for x in feat_names if x not in ["brouwer_index", "flesch_douma", "flesch_reading", "mcalpine"]]
        else:
            feat_names = [x for x in feat_names]

    # Compute feature importances if attribute exists
    if hasattr(model, "feature_importances_"):
        # Use estimators for tree models
        if hasattr(model, "estimators_"):
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        else:
            std = np.std(model.feature_importances_)
            # std = np.zeros(len(feat_names))

        result_df = pd.DataFrame({
            'Feature': feat_names, #forest_importances.index,
            'Importance': model.feature_importances_.round(5) ,#forest_importances.values.round(5),
            'Std Deviation': std.round(5)
        }).sort_values(by=["Importance"], ascending=False)

        print("Feature Importances:")
        print(result_df)

        plt.figure(figsize=(20, 10))
        fig, ax = plt.subplots()
        ax.bar(
            result_df["Feature"],
            result_df["Importance"],
            yerr=result_df["Std Deviation"],
        )
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        plt.xticks(rotation=90, ha="center")
        fig.tight_layout()
        plt.savefig(f"{args.figure_save}feature_importance_mdi.png")

    """
    perm_result = permutation_importance(
        model, x_test, y_test, n_repeats=5, random_state=args.random_seed, n_jobs=-1
    )

    perm_df = pd.DataFrame({
        'Feature': feat_names, #forest_importances.index,
        'Importance': perm_result.importances_mean.round(5),
        'Std Deviation': perm_result.importances_std.round(5)
    }).sort_values(by=["Importance"], ascending=False)

    print("Permutation Importances:")
    print(perm_df)

    plt.figure(figsize=(20, 10))
    fig, ax = plt.subplots()
    ax.bar(
        perm_df["Feature"],
        perm_df["Importance"],
        yerr=perm_df["Std Deviation"],
    )
    ax.set_title("Permutation Importances")
    ax.set_ylabel("Decrease in score (R^2)")
    plt.xticks(rotation=90, ha="center")
    fig.tight_layout()
    plt.savefig(f"{args.figure_save}feature_importance_perm.png")
    
    """

    logger.info(f"Model: {model}")
    # !Add checker or model instance/type? !
    explainer = shap.Explainer(model, x_test)
    print(f"Explainer baseline prediction: {explainer.expected_value}")
    shap_values = explainer(x_test, check_additivity=False)

    # Compute mean absolute SHAP values
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_std = np.std(shap_importance)

    shap_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance': shap_importance,
        'Std Deviation': shap_std.round(5)
    }).sort_values(by='Importance', ascending=False)

    print("Mean Shap Value (Importance):")
    print(shap_df)

    plt.figure(figsize=(20, 10))
    shap.summary_plot(shap_values, features=x_test, feature_names=feat_names)
    plt.savefig(f"{args.figure_save}feature_importance_shap_bees.png")

    plt.figure(figsize=(20, 10))
    shap.summary_plot(shap_values, features=x_test, feature_names=feat_names, plot_type="bar")
    plt.savefig(f"{args.figure_save}feature_importance_shap_mean.png")



if __name__ == "__main__":
    main()
