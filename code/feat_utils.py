import numpy as np
import textstat
from datasets import Dataset
import os
from minicons import scorer
from minicons.scorer import IncrementalLMScorer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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


def avg_sen_len(fragment: str) -> float:
    """ """
    # The commented return statement returns the avg character length per sentence
    # return np.mean([len(sentence) for sentence in fragment.split('.')[:-1]])

    # '[:-1]' is needed to remove the tailing empty string,
    lens = [len(sentence.split()) for sentence in fragment.split(".")[:-1]]
    # Control for one sentence without full stop --> if statement
    return np.mean(lens) if lens else 0.0


def syll_count(fragment: str) -> float:
    # Uses Pyphen for syllable calculation
    try:
        return textstat.syllable_count(fragment)
    except:
        return 0.0


def brouwer_index(asl: float, ans: float) -> float:
    """
    Compute Brouwer's Index, and normalise to float between 0 and 1.

    :param asl: Average sentence length of the text fragment.
    :param ans: Average number of syllables in the text fragment.
    :return:
    """
    try:
        # A score between 0 and 120, where 0 represents difficult and 120 easy (negative slope)
        score = 195 - (2 * asl) - (67 * ans)
        # Normalise to a float between 0 and 1
        return score / 120
    except:
        return 0.0


def flesch_douma(asl: float, ans: float) -> float:
    """
    Compute the Flesch-Douma score, and normalise to float between 0 and 1.

    :param asl: Average sentence length of the text fragment.
    :param ans: Average number of syllables in the text fragment.
    :return:
    """
    try:
        # A score between 0 and 120, where 0 represents difficult and 120 easy (negative slope)
        score = 207 - (.93 * asl) - (77 * ans)
        # Normalise to a float between 0 and 1
        return score / 120
    except:
        return 0.0


def load_surp_model(hf_model: str, hf_token: str) -> IncrementalLMScorer:
    """ """
    # Special thanks to: https://github.com/kanishkamisra/minicons and
    # https://kanishka.website/post/minicons-running-large-scale-behavioral-analyses-on-transformer-lms/

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token
    )

    surp_model = scorer.IncrementalLMScorer(
        model=model,
        tokenizer=tokenizer,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    return surp_model


def comp_read(example: dict[str, any]) -> dict[str:str]:
    """
    Calculates readability formulae and processes them into the Dataset object.

    :param example: Each row (data point/sample) of the Dataset object.
    :return: example: dictionary with the added readability metrics.
    """

    example["syll_count"] = str(syll_count(example["body"]))
    example["avg_syll_tok"] = str(int(example["syll_count"]) / int(len(example["body"].split(" "))))
    example["brouwer_index"] = str(brouwer_index(float(avg_sen_len(example["body"])), float(example["avg_syll_tok"])))
    example["flesch_douma"] = str(flesch_douma(float(avg_sen_len(example["body"])), float(example["avg_syll_tok"])))
    #example["brouwer_index_mor"] = str(brouwer_index(float(avg_sen_len(example["body"])), float(example["Morf_per_wrd"])))
    #example["flesch_douma_mor"] = str(flesch_douma(float(avg_sen_len(example["body"])), float(example["Morf_per_wrd"])))

    return example


