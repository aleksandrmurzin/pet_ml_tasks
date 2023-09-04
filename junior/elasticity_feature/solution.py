import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression as LR


def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the elasticity for each SKU in a given DataFrame.

    Args:
    df (pd.DataFrame): A DataFrame containing at least the following columns:
                      - sku: SKU identifier
                      - qty: Quantity sold
                      - price: Price of the product

    Returns:
    pd.DataFrame: A DataFrame with two columns:
                 - sku: SKU identifier
                 - elasticity: Elasticity score for each SKU
    """

    def fit_predict(P, q):
        """
        Fit a linear regression model to predict quantity based on price.

        Args:
        P (pd.Series): Series containing product prices.
        q (pd.Series): Series containing product quantities.

        Returns:
        pd.Series: Predicted quantities based on the linear regression model.
        """
        P = P.values.reshape(-1, 1)
        model = LR()
        model.fit(P, q)
        preds = model.predict(P)
        return preds

    def calc_metric(q, preds):
        """
        Calculate the R-squared score as a metric.

        Args:
        q (pd.Series): Actual quantities.
        preds (pd.Series): Predicted quantities.

        Returns:
        float: R-squared score indicating the goodness of fit.
        """
        return r2_score(q, preds)

    result = df.groupby(by=["sku"]).apply(
         lambda x: calc_metric(
              np.log(x["qty"]+1), fit_predict(x["price"],
              np.log(x["qty"]+1))))
    result = result.reset_index(drop=False)
    result.columns = ["sku", "elasticity"]

    return result
