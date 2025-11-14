""" Utility functions used for the regression analysis """
import numpy as np
import pandas as pd


def get_data(file_path: str, dv: str) -> tuple[np.array, np.array]:
    """
    Read Dataframe, and return the independent variables as 'x' and the dependent variable as 'y'.
    Args:
        file_path (str): Path to the file containing the data.
        dv (str): The dependent variable to use.
    Returns:
        np.array, np.array: A np.array with the predictor values, and one with the response variable values.
    """
    file_df = pd.read_json(file_path, lines=True)
    return file_df.drop(dv, axis=1), file_df[dv]
