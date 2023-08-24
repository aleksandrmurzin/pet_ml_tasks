from utils import GradientBoostingRegressor, mse, mae
import pandas as pd


data = pd.read_csv("data.csv")
X, y = data.drop(columns=["delay_days"]), data["delay_days"]

gbr = GradientBoostingRegressor()
gbr.fit(X, y)
y_pred = gbr.predict(X)

def test_GradientBoostingRegressor_fit():
    assert y_pred[0] == y.mean()



def test_mse():
    mse(y, y_pred)
    mae(y, y_pred)