from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


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
        subsample_size=0.5,
        replace=False
    ):
        self.n_estimators = n_estimators
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.loss=loss
        self.verbose=verbose
        self.trees_ = []
        self.subsample_size = subsample_size
        self.replace = replace


    def _subsample(self, X, y):
        size = int(len(X) * self.subsample_size)
        idx = np.random.choice(len(X), size=size, replace=self.replace)
        sub_X = X[idx,:]
        sub_y = y[idx]
        return sub_X, sub_y

    def _mae(self, y_true, y_pred) -> Tuple[float, np.ndarray]:
        """Mean absolute error loss function and gradient."""
        loss = np.sum(np.abs(y_pred - y_true)) / len(y_true)
        grad = np.sign(y_pred - y_true)
        return loss, grad

    def _mse(self, y_true, y_pred) -> Tuple[float, np.ndarray]:
        """Mean square error loss function and gradient."""
        loss = np.sum(np.square(y_pred-y_true)) / len(y_true)
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
        preds = self.base_pred_ = y.mean() # first pred

        while iterator_n_estimator < self.n_estimators:
            if self.loss == "mse": # calc gradient
                loss, residuals = self._mse(y, preds)
                if self.verbose:
                    print(loss)

            elif self.loss == "mae":
                loss, residuals = self._mae(y, preds)

            else:
                loss, residuals = self.loss(y, preds)
            antigradient = -residuals

            dtr = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_split,)
            dtr.fit(*self._subsample(X, antigradient))
            preds +=  dtr.predict(X) * self.learning_rate
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
        preds = self.base_pred_
        for tree in self.trees_:
            preds +=  tree.predict(X) * self.learning_rate
        return preds
