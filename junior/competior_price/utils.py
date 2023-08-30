import numpy as np
import pandas as pd


def agg_comp_price(X_init: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    X_init : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    def apply_agg(x, y, r):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        y : _type_
            _description_
        r : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if x == "max":
            return np.max(y)
        elif x == "min":
            return np.min(y)
        elif x == "med":
            return np.median(y)
        elif x == "avg":
            return np.mean(y)
        elif x == "rnk":
            if r == -1:
                return np.nan
            return y[r]

    def comp_price(x, y):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        y : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if y != y:
            return x
        if y / x < 1.2 and y / x > 0.8:
            return y
        return x

    X = X_init.copy()
    X = X.groupby(by=["sku"]).agg({
        "rank": "min",
        "agg": "first",
        "base_price": "max",
        "comp_price": list
    })
    X = X.reset_index(drop=False)

    X["comp_price"] = X.apply(lambda x:  apply_agg(
        x['agg'], x["comp_price"], x["rank"]), axis=1)
    X["new_price"] = X.apply(lambda x: comp_price(
        x["base_price"], x["comp_price"]), axis=1)
    X = X.drop(columns=['rank'])
    return X
