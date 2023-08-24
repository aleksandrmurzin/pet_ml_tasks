import numpy as np
import pandas as pd
from typing import Tuple

# class GradientBoostingRegressor:
#     """Gradient boosting regressor."""

#     def fit(self, X, y):
#         """Fit the model to the data.

#         Args:
#             X: array-like of shape (n_samples, n_features)
#             y: array-like of shape (n_samples,)

#         Returns:
#             GradientBoostingRegressor: The fitted model.
#         """
#         self.base_pred_ = y.mean()



#     def predict(self, X):
#         """Predict the target of new data.

#         Args:
#             X: array-like of shape (n_samples, n_features)

#         Returns:
#             y: array-like of shape (n_samples,)
#             The predict values.

#         """
#         return np.full(shape=(len(X)), fill_value=self.base_pred_)



# def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
#     """Mean squared error loss function and gradient."""

#     loss = np.sum(np.square(y_pred - y_true)) / len(y_true)
#     grad = y_pred - y_true
#     return loss, grad

# def mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
#     """Mean absolute error loss function and gradient."""

#     loss = np.sum(np.abs(y_pred - y_true)) / len(y_true)
#     grad = np.sign(y_pred - y_true)
#     return loss, grad



import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
    ):
        # YOUR CODE HERE
        ...

    def _mse(self, y_true, y_pred):
        # YOUR CODE HERE
        ...

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.base_pred_ = y.mean()
        def rec(self):
            return None
        ...

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """
        # YOUR CODE HERE
        ...
        return predictions
