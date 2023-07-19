import pandas as pd
import numpy as np


def fillna_with_mean(
    df: pd.DataFrame, target: str, group: str
) -> pd.DataFrame:
    """
    Fill missing values in a DataFrame column with the mean value of the corresponding group.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        target (str): The name of the column with missing values to be filled.
        group (str): The name of the column used for grouping the data.

    Returns:    
        pd.DataFrame: The DataFrame with missing values filled using the mean value of the group.

    """
    df_result = df.copy()
    df_result[target] = df_result.groupby(by=group)[target]\
        .transform(lambda x: x.fillna(x.mean()))\
        .apply(np.floor)
    return df_result
