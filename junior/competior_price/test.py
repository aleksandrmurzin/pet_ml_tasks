from utils import agg_comp_price
import pandas as pd

data = pd.read_csv("data.csv", sep="\t")


def test_agg_comp_price():
    agg_comp_price(data)