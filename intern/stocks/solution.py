import pandas as pd
import numpy as np


def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limits the GMV (Gross Merchandise Value) in a DataFrame 
    based on the product price and stock quantity.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing columns 'gvm',
          'price', and 'stock'.

    Returns:
        pd.DataFrame: The modified DataFrame with the 'gvm'
         values limited by the product price and stock quantity.
    """

    df_results = df.copy()
    
    mask = df["gmv"] / df_results["price"] > df_results["stock"]
    df_results.loc[mask, "gmv"] = df_results["price"] * df_results["stock"]
    df_results.loc[~mask, "gmv"] = df_results["price"] * np.floor_divide(df_results["gmv"],
                                                                         df_results["price"])
    return df_results
