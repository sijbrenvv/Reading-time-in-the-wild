import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import set_seed
import logging
import os
import warnings

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
        "--data",
        "-d",
        required=True,
        help="Path to the data file.",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        "-out",
        required=True,
        help="Path where to save the data splits.",
        type=str,
    )
    parser.add_argument(
        "-dp",
        "--dependent_variable",
        type=str,
        default="avg_rt_tok",
        help="The dependent/response variable. Default='avg_rt_tok'.",
    )
    parser.add_argument(
        "--days_of_week",
        "-dow",
        help="The days of the week to include: weekend, weekdays, or all. Default='all'.",
        choices=["weekend", "weekdays", "all"],
        default="all"
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
    parser.add_argument(
        "--extra_feats",
        "-ef",
        nargs="*",
        help="The extra features to add to the specified dimension/block of features. For example: -ef/--extra_feats feat1 feat2.",
        default=[]
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = create_arg_parser()

    # Set seed for replication
    set_seed(args.random_seed)

    # Read the data
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file '{args.data}' not found.")
    logger.info("Loading and processing data...")
    data_df = pd.read_csv(args.data, sep=",")

    feature_block = {
        #"tscan": ["wrd_freq_log_zn_corr", "wrd_freq_zn_log", "Conc_nw_ruim_p", "Conc_nw_strikt_p", "Alg_nw_d",
        #          "Pers_ref_d", "Pers_vnw_d", "Wrd_per_zin", "Wrd_per_dz", "Inhwrd_dz_zonder_abw", "AL_max",
        #          "Bijzin_per_zin", "Bijv_bep_dz_zbijzin", "Extra_KConj_dz", "MTLD_inhwrd_zonder_abw"],
        "read": ["brouwer_index", "flesch_douma", "LiNT_score1", "LiNT_score2"],
        "llm": ["surp_fietje-2", "surp_EuroLLM-9B", "surp_mGPT", "surp_tweety-7b-dutch-v24a",
                "surp_Qwen3-4B", "surp_Qwen3-1.7B"],
        "metadata": ["domain", "sentiment_label_detailed", "classification", "day_of_week"],
        #"prof-ud": ["tokens_per_sent", "lexical_density", "avg_verb_edges", "verbal_root_perc", "avg_max_depth",
        #            "avg_token_per_clause", "avg_prepositional_chain_len", "avg_max_links_len", "upos_dist_PROPN",
        #            "principal_proposition_dist"]
    }

    # Select columns
    # Drop the features we do not want to use as predictors
    #data_df = data_df.drop(["title", "publication_date", "body", "views", "read_time_sec",
    #                        "avg_rt_view", "baseline_rt", "n_sentences", "n_tokens", "syll_count", "word_count_category",
    #                        "mono_syll_count", "poly_syll_count", "n_prepositional_chains", "domain", "classification",
    #                        "brouwer_index", "flesch_douma", "flesch_reading", "mcalpine"], axis=1)
    if args.block_of_feature == "prof-ud" or args.block_of_feature == "tscan":
    #    data_df = data_df.drop(["title", "publication_date", "body", "views", "read_time_sec",
    #                            "avg_rt_view", "baseline_rt", "domain", "classification", "word_count_category",
    #                            "sentiment_label_detailed"], axis=1)
        data_df = data_df.drop(["title", "publication_date", "body", "views", "read_time_sec", "Inputfile",
                                "avg_rt_view", *feature_block["llm"], *feature_block["metadata"],
                                *feature_block["read"], "syll_count", "avg_syll_tok"], axis=1)  # , "syll_count", "avg_syll_tok"
    elif args.block_of_feature in {"read", "llm", "metadata"}:
        data_df = data_df[feature_block[args.block_of_feature] + [args.dependent_variable]]
    elif args.block_of_feature == "tscan" or args.block_of_feature == "prof-ud" and args.extra_feats:
        other_dims = [*feature_block["metadata"], *feature_block["read"], *feature_block["llm"]]
        col_to_drop = [feat for feat in other_dims if feat not in args.extra_feats]
        data_df = data_df.drop(["title", "publication_date", "body", "views", "read_time_sec", "Inputfile",
                                "avg_rt_view", "syll_count", "avg_syll_tok", *col_to_drop], axis=1)  # , "syll_count", "avg_syll_tok"

                            #"brouwer_index", "flesch_douma", "flesch_reading", "mcalpine"], axis=1)
    #data_df = data_df.drop(["title", "publication_date", "body", "avg_rt_view", "baseline_rt"], axis=1)

    # Move dependent variable to final position
    dv_col = data_df.pop(args.dependent_variable)
    data_df.insert(loc=len(data_df.columns), column=args.dependent_variable, value=dv_col)
    # data_df = data_df[["n_sentences", "tokens_per_sent", "char_per_tok", "avg_lexical_chain_len_tok", "hapax_rich", args.dependent_variable]]

    if args.days_of_week == "weekend":
        data_df = data_df[data_df["day_of_week"].isin([5, 6])]

    elif args.days_of_week == "weekdays":
        data_df = data_df[data_df["day_of_week"].isin([0, 1, 2, 3, 4])]

    logger.info(f"Data dataframe shape: {data_df.shape}")

    # Write kept features to output
    with open(f"{args.output_file}features.txt", "w") as file:
        file.write("\n".join(map(str, data_df.columns)))
    logger.info("Stored used features in output folder.")

    # No shuffle needed as train_test_split will shuffle by default
    train_df, test_df = train_test_split(
            data_df,
            test_size=0.2,
            random_state=args.random_seed,
        )

    # Output splits to the predefined folder
    logger.info("Outputting splits to predefined folders...")
    os.makedirs(args.output_file, exist_ok=True)
    train_df.to_json(os.path.join(args.output_file, "train.json"), orient="records", lines=True)
    logger.info(f"Train split saved to: {os.path.join(args.output_file, 'train.json')}")
    test_df.to_json(os.path.join(args.output_file, "test.json"), orient="records", lines=True)
    logger.info(f"Validation split saved to: {os.path.join(args.output_file, 'test.json')}")