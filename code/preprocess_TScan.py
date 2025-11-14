import argparse
import logging
import os
import pandas as pd
from datasets import Dataset
from transformers import set_seed
import warnings
import re

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
        # Convert categorical meta features to numeric
        return Dataset.from_pandas(factorize(pd.read_csv(inp_file)))
        # return Dataset.from_csv(inp_file)
    elif inp_file.endswith(".json") or inp_file.endswith(".jsonl"):
        # Convert categorical meta features to numeric
        return Dataset.from_pandas(factorize(pd.read_json(inp_file, lines=True)))
        # return Dataset.from_json(inp_file)


def factorize(df: pd.DataFrame) -> pd.DataFrame:
    """"""

    cat_columns = ["sentiment_label_detailed", "classification", "domain"]
    # Convert categorical variables to numeric
    #df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0] + 1)
    #return df

    mappings = {}

    for col in cat_columns:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes + 1
        # Map number to value
        mappings[col] = dict(enumerate(uniques, start=1))

    # Store mappings
    with open("data/col_mappings.txt", "w", encoding="utf-8") as f:
        for col, mapping in mappings.items():
            f.write(f"Column: {col}\n")
            for code, label in mapping.items():
                f.write(f"{code}: {label}\n")
            f.write("\n")

    return df


def get_sentences(text: str) -> list[str]:
    """"""

    sents = []

    # 1. Mark sentence boundary
    text = re.sub(
        r'([.!?;])(\s+)(?=[A-Z0-9"\'])',
        r'\1^\2',
        text
    )

    # 2. Mark glued subheaders / listings
    text = re.sub(
        r'([a-zà-ÿ])([A-Z0-9])',
        r'\1@ \2',
        text
    )

    # 3. Append @ to standalone URLs (preceded by punctuation and space)
    text = re.sub(
        r'([.!?;])\s+((https?://|www\.)[^\s]+)',
        r'\1 @\2',
        text
    )

    # Split on caret and get sentences
    temp_sents = [s.strip() for s in text.split('^') if s.strip()]

    # Search for subheaders, listings, and standalone urls, and prepend them with "###"
    for sent in temp_sents:

        sp_sen = sent.split("@")
        if len(sp_sen) > 1:
            if sp_sen[-1].startswith("www.") or \
            sp_sen[-1].startswith("https://") or \
            sp_sen[-1].startswith("http://") and \
            len(sp_sen[-1].split()) == 1:
                for i in range(len(sp_sen) - 1):
                    sents.append(sp_sen[i])
                sents.append(f"### {sp_sen[-1]}")
            else:
                for i in range(len(sp_sen) - 1):
                    sents.append(f"### {sp_sen[i]}")
                sents.append(sp_sen[-1])
        else:
            sents.append(sp_sen[0])

    # Start with empty line (newline) if doc starts with skip marker
    if sents[0].startswith("### "):
        sents[0] = "\n" + sents[0]

    #logger.info(f"Sents: {sents}")
    #logger.info(f"Texts: {text}")

    return sents


def preprocess_for_tscan(example: dict[str, any]) -> dict[str:str]:
    """
    Full preprocessing pipeline for T-Scan:
    - replaces square brackets
    - normalizes quotes
    - adjusts spacing
    - moves closing quote before punctuation
    - dynamically marks subheaders, bullets, URLs, etc. as ### comments

    :param example: Each row (data point/sample) of the Dataset object, including the text.
    :return: example: dictionary with the updated text.
    """

    text = example["body"]

    # Rule 1: Replace square brackets with round brackets
    text = re.sub(r'\[', '(', text)
    text = re.sub(r'\]', ')', text)

    # Rule 2: Normalise quotes and apostrophes
    # Replace all curly quotes with straight quotes
    # Replace double single quote with single double quote
    text = re.sub(r'[“”„«»]', '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"(''|,,)", '"', text)

    # Rule 3: Convert time notation like 12.00 to 12:00
    text = re.sub(
        r'\b([01]?[0-9]|2[0-3])\.([0-5][0-9])\b',
        r'\1:\2',
        text
    )

    # Rule 4a: Remove in-sentence punctuation inside quotes
    text = re.sub(
        r'(["\'])(.*?)([.!;])(["\'])(?=\s+[a-zà-ÿ])',
        r'\1\2\4',
        text
    )

    # Rule 4b: Move quotes at end of sentence inside punctuation
    text = re.sub(
        r'([.!;?])(["\'])',
        r'\2\1',
        text
    )

    # Rule 5: Add space after punctuation (excluding commas)
    # if followed by uppercase, digit, quotation mark, or url.
    text = re.sub(
        r'([.!?;])(?=([A-Z0-9"\']|www\.|https?://))',
        r'\1 ',
        text
    )

    # Rule 6: Remove hard returns mid-sentence
    lines = text.splitlines()
    new_lines = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                new_lines.append(buffer.strip())
                buffer = ""
            new_lines.append("")
        else:
            if buffer:
                if not re.search(r"[.;!?]['\"]?\s*$", buffer):
                    buffer += " " + stripped
                else:
                    new_lines.append(buffer.strip())
                    buffer = stripped
            else:
                buffer = stripped
    if buffer:
        new_lines.append(buffer.strip())

    text = "\n".join(new_lines)

    # Rule 7a: Avoid list symbols ending with periods
    text = re.sub(r"([-•*])\.", r"\1 ", text)
    #logger.info(f"Text Rule 6: {text}")

    # Rule 7b: Replace numeric list prefixes such as "1." at line start with colon
    text = re.sub(
        r'(?m)^(\d{1,2})\.(?=\s+\S)',
        r'\1:',
        text
    )

    # Rule 7c: Replace inline numbered lists such as "1. Mix de ingrediënten, 2. Bak het 10 minuten" with colons
    text = re.sub(
        r'(?<!-)\b(\d{1,2})\.(?=\s+[A-Za-zÀ-ÿ])',
        r'\1:',
        text
    )

    # Rule 8: Add space between € and number
    text = re.sub(r"€(\d)", r"€ \1", text)

    # Rule 9: Remove continuing text marker
    text = re.sub(r"\([Tt]ekst loopt door (onder|na) de foto'?s?\)", " ", text)

    # Rule 10: Remove redundant whitespace
    text = re.sub(r"[ \t]+", " ", text)

    sen_list = get_sentences(text)
    example["body"] = "\n".join(sen_list)

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

    #dataset = dataset.rename_columns({"article_body": "body", "read_time": "read_time_sec", "publish_date": "publication_date"})
    #logger.info(f"dataset columns: {dataset.column_names}")

    # Remove instances with an empty body and no reading time
    # Remove articles with fewer than 100 words
    logger.info(f"Apply data selection requirements...")
    dataset = dataset.filter(lambda example: example["body"] is not None)
    dataset = dataset.filter(lambda example: example["read_time_sec"] is not None)
    dataset = dataset.filter(lambda example: example["read_time_sec"] != 0)
    dataset = dataset.filter(lambda example: example["views"] >= 25)
    dataset = dataset.filter(lambda example: example["publication_date"].startswith("2024"))
    dataset = dataset.filter(lambda example: len(example["body"].split(" ")) >= 100)

    # Remove duplicate entries
    temp_df = dataset.to_pandas()
    temp_df = temp_df.sort_values(by=["publication_date", "views"], ascending=[True, False])
    temp_df = temp_df.drop_duplicates(subset=["publication_date"], keep="first")
    temp_df = temp_df.drop_duplicates(subset=["title"], keep="first")
    temp_df = temp_df.drop_duplicates(subset=["body"], keep="first")
    day_of_week = pd.to_datetime(temp_df["publication_date"]).dt.weekday
    temp_df.insert(loc=1, column="day_of_week", value=day_of_week)
    temp_df["id"] = [i for i in range(temp_df.shape[0])]
    dataset = Dataset.from_pandas(temp_df)
    del temp_df

    ## Use a subset for testing purposes
    #dataset = dataset.select(range(200))

    #body= "De provincie is tegen gaswinning. Fryslân vindt dat het plan onvoldoende inzicht geeft over de gevolgen van de verruimde winning van gas. Vier veldjesOp 16 mei 2022 heeft Vermilion Energy het ministerie gevraagd mee te werken aan een gewijzigd winningsplan De Blesse-Blesdijke. In dat plan staat wat er verandert bij de winning uit de vier veldjes De Blesse, De Blesse East, Blesdijke East en Blesdijke. Sinds 1999 wint Vermilion al gas uit de velden De Blesse en Blesdijke. Met een afgetakte boring is daar in 2021 het veld De Blesse East bijgekomen. Meer en langerVermilion vraagt voor de velden Blesdijke East en De Blesse een verlenging van de periode dat gas gewonnen mag worden met drie jaar tot en met 2039. Ook wordt de hoeveelheid gas dat gewonnen gaat worden hoger. Voor het gasveld De Blesse is het productievolume naar beneden bijgesteld en uit het gasveld Blesdijke wordt de productie niet hervat. Er is sprake van onzekerheid in dit winningsplan, omdat de grootte van het gasvoorkomen De Blesse East nog niet is aangetoond. Deze extra boring moet nog worden uitgevoerd. Aanvullende gegevensDe staatssecretaris wil daarom wel dat Vermilion daarna aanvullende gegevens verstrekt over de omvang van de productie en de gevolgen voor de bodemdaling. Dan wordt duidelijk of eerdere berekeningen juist waren. Desondanks heeft de staatssecretaris ingestemd met het gewijzigde winningsplan, omdat het voldoet aan de wettelijke eisen. BeroepHet instemmingsbesluit, het winningsplan en alle bijbehorende documenten lagen sinds 30 november tot en met woensdag 10 januari ter inzage. In die periode kon beroep worden aangetekend bij de afdeling bestuursrechtspraak van de Raad van State. De provincie Fryslân legt zich dus niet neer bij het besluit van de staatssecretaris. Zorgen over Koloniën van WeldadigheidDe provincie is niet bang dat de veiligheid van bewoners in het geding is, maar zorgen zijn er wel over mogelijke gevolgen voor natuur en milieu en de Koloniën van Weldadigheid. Zo wordt nu door Vermilion een bodemdaling voorspelt van vijf tot acht centimeter. Dat was vijf centimeter. ""Dat dit zijn weerslag zal hebben op zowel natuur als milieu lijkt evident"", laat de provincie weten. En hoe klein ook, de theoretische kans op een aardbeving is nu ook iets groter geworden. InventarisatieDe gemeenten Steenwijkerland en Weststellingwerf zijn ook tegen de verruiming van de winning. Weststellingwerf heeft precies een jaar geleden gepleit voor een inventarisatie van de kwetsbare objecten in het winningsgebied, een monitoring van de huidige bouwkundige status en een onderzoek naar de impact van de gaswinning op deze objecten. De staatssecretaris volstaat met de conclusie dat Vermilion verantwoordelijk is voor alle schade die zou kunnen ontstaan."
    #body = "IngrediëntenVoor 4 personen:1 eetlepel korianderzaadjes, gevijzeld1/4 theelepel gemalen kardemom1 gele ui, geschild en in grove stukken1 duim gember, geschild2 teentjes knoflook, geschild1/2 eetlepel komijnzaadjes, gevijzeldPlantaardige olie400 gram baby spinazie1/2 limoen, het sap1 groene chilipeper1 theelepel zout1 theelepel komijnpoeder240 ml kokosmelk (1 groot blik)200 gram paneer, in dobbelsteentjesSnuf chilipoederRijst en/of naan voor erbijZo maak je hetDoe de korianderzaadjes met de kardemom, gele ui, gember en knoflook in een hakmolentje en hak tot een grove pasta. Verhit een scheut plantaardige olie in een grote pan en voeg de pasta toe. Roer met een houten lepel tot alle aroma's vrijkomen. Voeg de babyspinazie (dit lukt het beste in delen) toe en roer tot de spinazie geslonken is. Doe zodra dat het geval is het limoensap, de chilipeper en het zout erbij en roer en mix met je staafmixer tot een gladde, dikke massa. Voeg het komijnpoeder, kokosmelk en de paneer toe en roer opnieuw. Laat de kokos saag paneer nog 5 minuutjes op laag vuur pruttelen zodat de kaas mee verwarmt. Serveer de kokos paneer met rijst en/of naan."
    #body = "Je hebt nodigVoor 4-6 personen3 aardappelen, geschild en in blokjes8 papadums25 gram gepofte rijst400 gram kikkererwten (Bonduelle)2 theelepels komijn1 mango, in blokjes1 komkommer, in blokjes1 rode ui, fijngehaktDe pitjes van 1 granaatappelHandje koriander, fijngehakt250 gram Griekse yoghurt (of een plantaardig alternatief)4 – 6 eetlepels mango-chutneyEventueel wat sevOlijfolieZout en peperVoor de dressing1 eetlepel chaat masala of garam masala2 eetlepels tamarinde240 ml waterZo maak je deze bhel puriVerwarm de oven voor op 230 graden. Spoel de kikkererwten af, dep ze droog en leg ze op een bakplaat. Meng met de komijn, wat olijfolie en zout en peper. Zet ze daarna 10 tot 15 minuten in de oven, tot ze knapperig zijn. Maak ondertussen de dressing: meng de tamarinde met het water in een steelpannetje, breng aan de kook en laat ongeveer tien minuten pruttelen. De saus moet tot de helft indikken, dan is 'ie goed. Dit geeft je tijd om de aardappelen aan de kook te brengen. Vijf tot tien minuten zou genoeg moeten zijn. Rooster dan de chaat masala of garam masala kort (een minuutje) in een droge koekenpan, zodat de smaken loskomen. Voeg toe aan de tamarindedressing en zet die opzij. Nu kun je de salade gaan opmaken. Meng de komkommer, de mango en de granaatappelpitjes met de aardappelblokjes, stukjes papadum, gepofte rijst en sev. Bestrooi met de krokante kikkererwten, wat fijngehakte rode ui en koriander. Schep hier wat van de tamarindedressing doorheen. Maak af met een dikke dot yoghurt en wat mangochutney. Je kunt de bhel puri lekker opscheppen met stukjes papadum.www.culy.nl"
    #body = "Ook voor woensdag wordt er nog een onstuimige dag verwacht met in de kustprovincies een harde wind en kans op zware windstoten. Aan de noordwestkust stormt het tijdelijk. In de ochtend neemt de wind af. Er zijn enkele buien en in de middag is vooral in het zuiden kans op een pittigere bui met onweer. Het wordt 9 tot 11 graden. In verschillende delen van het land was dinsdagavond de brandweer uitgerukt om bomen die door de harde wind zijn omgewaaid weg te halen. Ook het treinverkeer was op een aantal trajecten verstoord. Rijkswaterstaat waarschuwde weggebruikers om rekening te houden met gevaarlijke situaties. Code oranjeCode oranje gold sinds 22.00 uur voor de provincies Noord-Holland en Friesland en voor het IJsselmeer en het Waddengebied"". In het noordelijk kustgebied werden (zeer) zware windstoten gemeten 80-110 km/uur. Elders kwamen zware windstoten voor van 75-90 km/uur"", aldus het KNMI. Op de Harlingerstraatweg in Leeuwarden belandde een auto in het water. Daarbij raakte niemand gewond. De brandweer in Meppel was druk met een overstroming van een parkeerterrein en van enkele tuinen en garages aan het Vliegenpad. Treinen en dijkenOok in het zuiden van het land ontstond schade door de storm. In Zeeland waaide een boom om op de N654 bij Noordgouwe, in Vlissingen liet de wandplaat van een huis los en waaide een plaat op een terras los. In Den Haag raakten vijf auto's beschadigd door een tak die van een boom afbrak en waaiden stellages los, aldus Omroep West. Op het treintraject tussen Utrecht Centraal en Tiel viel dinsdagavond een boom op het spoor. Daardoor reden er even geen treinen. Ook richting Rotterdam reden tijdelijk geen treinen door omgewaaide bomen. De N307, de dijk tussen Lelystad en Enkhuizen, was enige tijd afgesloten door een storing bij de Houtribsluizen in Lelystad. De weg ging even voor middernacht weer open."
    #temp = preprocess_for_tscan({"body": body})

    # Clean the dataset
    dataset = dataset.map(
        preprocess_for_tscan,
        remove_columns=["__index_level_0__", "article_type"]
    )

    # Export dataset
    logger.info(f"Exporting the updated dataset...")
    # os.makedirs(args.output_file_path, exist_ok=True)
    if args.output_file_path.endswith(".csv"):
        dataset.to_csv(args.output_file_path, index=False, sep=",")
        logger.info(f"Dataset saved to: {args.output_file_path}")
    if args.output_file_path.endswith(".json"):
        dataset.to_json(args.output_file_path, orient="records", lines=True)
        logger.info(f"Dataset saved to: {args.output_file_path}")
