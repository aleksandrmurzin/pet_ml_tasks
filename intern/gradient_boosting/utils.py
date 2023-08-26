import numpy as np
import pandas as pd
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

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



class GradientBoostingRegressor:
    """_summary_
    """
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.loss=loss
        self.verbose=verbose
        self.trees_ = []


    def _mae(self, y_true, y_pred) -> Tuple[float, np.ndarray]:
        """Mean absolute error loss function and gradient."""
        loss = np.sum(np.abs(y_pred - y_true)) / len(y_true)
        grad = np.sign(y_pred - y_true)
        return loss, grad

    def _mse(self, y_true, y_pred) -> Tuple[float, np.ndarray]:
        loss = np.sum(np.square(y_pred - y_true)) / len(y_true)
        grad = y_pred - y_true
        return loss, grad

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        iterator_n_estimator = 0

        self.base_pred_ = y.mean()

        if self.loss == "mse":
            _, grad = self._mse(y, self.base_pred_)
        elif self.loss == "mae":
            _, grad = self._mae(y, self.base_pred_)
        else:
            _, grad = self.loss(y, self.base_pred_)

        def fit_(X, grad):
            dtr = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_split)
            dtr.fit(X, grad)
            pred = dtr.predict(X)

            if self.loss == "mse":
                _, upd_grad = self._mse(grad, pred)
            elif self.loss == "mae":
                _, upd_grad = self._mae(grad, pred)
            else:
                _, upd_grad = self.loss(grad, pred)

            grad = grad + self.learning_rate * upd_grad
            return grad, dtr

        while iterator_n_estimator < self.n_estimators:
            grad, dtr = fit_(X, grad)
            self.trees_.append(dtr)
            iterator_n_estimator += 1
        return self

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """
        predictions = np.empty(len(X))
        for i in self.trees_:
            predictions += i.predict(X)
        return predictions
