from utils import GradientBoostingRegressor
import pandas as pd


data = pd.read_csv("data.csv")
X, y = data.drop(columns=["delay_days"]).values, data["delay_days"].values


def test_GradientBoostingRegressor():
    gbr = GradientBoostingRegressor(n_estimators=10000)
    gbr.fit(X, y)
    y_pred=gbr.predict(X)
    import pdb; pdb.set_trace()
