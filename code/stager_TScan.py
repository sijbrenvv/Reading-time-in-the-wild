import argparse
import logging
import os
import time
from typing import Optional
from dataclasses import dataclass
import requests
from datasets import Dataset
from transformers import set_seed
import warnings
from dotenv import load_dotenv
from tqdm import tqdm
from zipfile import ZipFile
import clam.common.client
import clam.common.status

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (username and password)
load_dotenv()

username = os.getenv("T-Scan_USR")
password = os.getenv("T-Scan_PAS")
address = "https://tscan.hum.uu.nl/tscan"


@dataclass
class Project:
    name: str
    status: int # 0 = staging 1 = scanning 2 = done
    statusmsg: str
    completion: int
    time: Optional[str] = None
    size: Optional[float] = None


def delete_input(project: str, name: str) -> bool:
    session = requests.Session()
    session.auth = (username, password)
    return session.delete(f"{address}/{project}/input/{name}.txt").status_code == 200


def delete_output(project: str) -> bool:
    session = requests.Session()
    session.auth = (username, password)
    return session.delete(f"{address}/{project}/output/").status_code == 200


def save_output_file(project: str, filename: str) -> None:
    session = requests.Session()
    session.auth = (username, password)
    response = session.get(f'{address}/{project}/output/{filename}')

    os.makedirs(os.path.join('exp/TScan_output', project), exist_ok=True)
    with open(os.path.join('exp/TScan_output', project, filename), 'wb') as target:
        target.write(response.content)


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
        "--random_seed",
        "-seed",
        #required=True,
        help="The random seed to use.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--project_name",
        "-pn",
        default="DvhN_rt_2024",
        help="How to name the project on T-Scan. Default = 'DvhN_rt_2024'",
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


if __name__ == "__main__":

    args = create_arg_parser()

    # Set seed for replication
    set_seed(args.random_seed)

    # Create and activate client -- connecting to server
    clamclient = clam.common.client.CLAMClient(address, username, password, basicauth=True)

    # Get the data
    logger.info(f"Loading the data...")
    dataset = get_data(args.input_file_path)

    # T-Scan analysis
    project_name = args.project_name

    # Create project
    clamclient.create(project_name)

    # Get project status and specification
    info = clamclient.get(project_name)
    #logger.info(f"Clam client info: {info}")

    # Directory for storing .txt files
    #os.makedirs("data/TScan_input", exist_ok=True)

    #logger.info(f"Creating zip file with inputs...")
    # Create .txt files
    #for entry in tqdm(dataset):
    #    with open(f"data/TScan_input/{str(entry['id'])}.txt", "w", encoding="utf-8") as file:
    #        file.write(entry["body"])

    # Create zip file/archive
    #zip_file = f"data/TScan_input/{project_name}.zip"
    #with ZipFile(zip_file, "w") as zipf:
    #    for i in range(len(dataset)):
    #        txt_f = f"data/TScan_input/{i}.txt"
    #        zipf.write(txt_f, arcname=f"{i}.txt")

    # Add input, scan input, get results, and delete input and output
    for c, entry in enumerate(tqdm(dataset)):
        if c > 8089:

            try:
                clamclient.addinputfile(project_name, info.inputtemplate("textinput"),f"data/TScan_input/{entry['id']}.txt")
            except:
                #delete_input(project_name, f"{entry['id'] - 1}")
                #delete_input(project_name, f"{entry['id']}")
                delete_output(project_name)
                delete_input(project_name, f"{entry['id']}")
                time.sleep(.1)
                clamclient.addinputfile(project_name, info.inputtemplate("textinput"), f"data/TScan_input/{entry['id']}.txt")
            time.sleep(.1)

            # Execute T-Scan
            info = clamclient.start(
                project_name,
                overlapSize='50',
                frequencyClip='99.0',
                mtldThreshold='0.72',
                useAlpino='yes',
                useWopr='no',
                sentencePerLine='no',
                prevalence='nl',
                word_freq_lex='subtlex_words.freq',
                lemma_freq_lex='freqlist_staphorsius_CLIB_lemma.freq',
                top_freq_lex='SoNaR500.wordfreqlist20000.freq'
            )

            # Check for parameter errors
            if info.errors:
                logger.info(f"An error occurred for file {entry['id']}.txt: {info.errormsg}")
                for parametergroup, paramlist in info.parameters:
                    for parameter in paramlist:
                        if parameter.error:
                            logger.info(f"Error in parameter {parameter.id}: {parameter.error}")
                # Delete input file and output files
                delete_output(project_name)
                delete_input(project_name, f"{entry['id']}")
                break

            # Wait for the project to be scanned
            while info.status != clam.common.status.DONE:
                time.sleep(20)
                # Get project status again
                info = clamclient.get(project_name)

            try:
                save_output_file(project_name, f"{entry['id']}.txt.document.csv")
            except Exception as e:
                logger.exception(f"Failed to process scan output for {entry['id']}.txt: {e}")
            # Delete input file and output files
            delete_output(project_name)
            delete_input(project_name, f"{entry['id']}")
            time.sleep(.1)

    time.sleep(1)
    exit()

