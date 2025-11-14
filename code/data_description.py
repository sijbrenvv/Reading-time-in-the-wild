import argparse
import pandas as pd
import numpy as np
import logging
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data(file_path: str) -> pd.DataFrame:
    """
    Function to read dataframe with columns.
    Args:
        file_path (str): .
    Returns:
        DataFrame object:.
    """

    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
        return pd.read_json(file_path, lines=True)


def data_description(data: pd.DataFrame) -> None:
    """"""
    logger.info(f"Describing the columns of '{input_path.split('/')[-1]}'...")
    logger.info(f"Data shape: {data.shape}")
    for col in data.columns:
        try:
            # Output standard error if dtype is numeric
            logger.info(f"\n{data[col].describe()}\nStandard Error: {data[col].sem()}")
        except TypeError:
            logger.info(f"\n{data[col].describe()}")

    logger.info(
        f"\n Domain unique: {pd.unique(data.domain)}\n Domain value counts: {pd.value_counts(data.domain).to_frame().reset_index()}")
    logger.info(
        f"\n Sentiment unique: {pd.unique(data.sentiment_label_detailed)}\n Sentiment value counts: {pd.value_counts(data.sentiment_label_detailed).to_frame().reset_index()}")
    logger.info(
        f"\n Classification unique: {pd.unique(data.classification)}\n Classification value counts: {pd.value_counts(data.classification).to_frame().reset_index()}")
    logger.info(
        f"\n DoW unique: {pd.unique(data.day_of_week)}\n DoW value counts: {pd.value_counts(data.day_of_week).to_frame().reset_index()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        "-inp",
        help="Path to the input data (json file). For example: 'data/dvhn_exp_data.json'.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--block_of_feature",
        "-bof",
        help="Select which block of features to analyse, including the dataset. Default: 'data'.",
        required=True,
        type=str,
        choices=["data", "prof-ud", "tscan", "read", "llm", "metadata"],
        default="data"
    )

    args = parser.parse_args()


    if not os.path.exists(args.input_file_path):
        raise FileNotFoundError(f"Input file '{args.input_file_path}' not found.")

    input_path = args.input_file_path

    # Get the data for the analyses
    logger.info(f"Loading the data...")
    data = get_data(input_path)

    feature_block = {
        "tscan": ["wrd_freq_log_zn_corr", "wrd_freq_zn_log", "Conc_nw_ruim_p", "Conc_nw_strikt_p", "Alg_nw_d",
               "Pers_ref_d", "Pers_vnw_d", "Wrd_per_zin", "Wrd_per_dz", "Inhwrd_dz_zonder_abw", "AL_max",
               "Bijzin_per_zin", "Bijv_bep_dz_zbijzin", "Extra_KConj_dz", "MTLD_inhwrd_zonder_abw",
                  "Inhwrd_d", "Spec_d", "Bvnw_d", "Nw_d", "Vnw_d", "avg_rt_tok"],
        "read": ["brouwer_index", "flesch_douma", "LiNT_score1", "LiNT_score2"],
        "llm": ["surp_fietje-2", "surp_EuroLLM-9B", "surp_mGPT", "surp_tweety-7b-dutch-v24a",
                "surp_Qwen3-4B", "surp_Qwen3-1.7B"],
        "metadata": ["domain", "sentiment_label_detailed", "classification", "day_of_week"],
        "prof-ud": ["tokens_per_sent", "lexical_density", "avg_verb_edges", "verbal_root_perc", "avg_max_depth",
                    "avg_token_per_clause", "avg_prepositional_chain_len", "avg_max_links_len", "upos_dist_PROPN",
                    "principal_proposition_dist", "upos_dist_ADJ", "upos_dist_NOUN", "upos_dist_PRON", "avg_rt_tok"]
    }

    if args.block_of_feature == "data":
        data_description(data=data)

    data = data[feature_block[args.block_of_feature]]
    for col in data.columns:
        try:
            # Output standard error if dtype is numeric
            logger.info(f"\n{data[col].describe()}\nStandard Error: {data[col].sem()}")
        except TypeError:
            logger.info(f"\n{data[col].describe()}")

    #logger.info(
    #    f"\n Domain unique: {pd.unique(data.domain)}\n Domain value counts: {pd.value_counts(data.domain).to_frame().reset_index()}")
    #logger.info(
    #    f"\n Sentiment unique: {pd.unique(data.sentiment_label_detailed)}\n Sentiment value counts: {pd.value_counts(data.sentiment_label_detailed).to_frame().reset_index()}")
    #logger.info(
    #    f"\n Classification unique: {pd.unique(data.classification)}\n Classification value counts: {pd.value_counts(data.classification).to_frame().reset_index()}")
    #logger.info(
    #    f"\n DoW unique: {pd.unique(data.day_of_week)}\n DoW value counts: {pd.value_counts(data.day_of_week).to_frame().reset_index()}")
