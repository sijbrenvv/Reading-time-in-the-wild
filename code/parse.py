import argparse
import logging
import os
import spacy_stanza
import spacy_udpipe
import stanza
from datasets import Dataset
from transformers import set_seed
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
        "--language",
        "-lang",
        default="nl",
        help="Set to 'multi' to enable automatic language detection.",
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


def download_spacy_udpipe(lang: str) -> spacy_udpipe:
    """
     Tries to download and load a Spacy-udpipe model for the specified language.
     Downloads and loads a Spacy-Stanza pipeline instead when any exceptions occur.

    :param lang: The specified language to load the spacy-udpipe/stanza model for.
    :return: spacy_udpipe: Loaded SpaCy-ud pipeline or \
    spacy_stanza.StanzaLanguage: Loaded Spacy-Stanza pipeline.
    """
    try:
        spacy_udpipe.download(lang)
        model = spacy_udpipe.load(lang)
        return model
    except Exception as e:
        stanza.download(lang)
        try:
            model = spacy_stanza.load_pipeline(lang)
            return model
        except Exception as e:
            model = spacy_stanza.load_pipeline("xx", lang='en')
            return model


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

    # Remove instances with an empty body
    #dataset = dataset.filter(lambda example: example["body"] != "")

    # Get the udpipe model and add conllu formatter
    logger.info("Loading the udpipe model...")
    nlp = download_spacy_udpipe(lang=args.language)
    config = {
        "include_headers": True,
        "conversion_maps": {"DEPREL": {"ROOT": "root"}}
                                       #"PUNCT": "punct"}}
    }
    nlp.add_pipe("conll_formatter", last=True, config=config)

    # Parse all the articles
    logger.info("Parsing the articles...")
    processed_data = dataset.map(
        lambda example: {**example, "conllu": nlp(example["body"])._.conll_str}
    )

    # Export dataset
    logger.info(f"Exporting the processed dataset...")
    if args.output_file_path.endswith(".csv"):
        processed_data.to_csv(args.output_file_path, index=False, sep=",")
        logger.info(f"Dataset saved to: {args.output_file_path}")
    if args.output_file_path.endswith(".json"):
        processed_data.to_json(args.output_file_path, orient="records", lines=True)
        logger.info(f"Dataset saved to: {args.output_file_path}")
