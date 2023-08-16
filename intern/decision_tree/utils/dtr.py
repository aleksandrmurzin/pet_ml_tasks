from __future__ import annotations
from dataclasses import dataclass

import numpy as np


class Node:
    """Decision tree node."""

    def __init__(self, feature=None, threshold=None, n_samples=None,
                 value=None, mse=None, left: Node = None, right: Node = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.n_samples = n_samples
        self.value = value
        self.mse = mse
        self.left = left
        self.right = right


@dataclass
class DecisionTreeRegressor:
    """Desision tree regressor"""

    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.array):
        """Compute the mean squared error of a vector."""
        mean_squared_error = np.mean(np.square(y - np.mean(y)))
        return mean_squared_error

    def _weighted_mse(self, y_left: np.array, y_right: np.array):
        weighted_mean_squared_error = (
            (self._mse(y_left) * len(y_left)
             + self._mse(y_right) * len(y_right))
            / (len(y_left) + len(y_right)))
        return weighted_mean_squared_error

    def _split(self, X: np.array, y: np.array, feature: int) -> tuple[float, float]:
        """Find the best split for a node (one feature)"""
        x = X[:, feature]
        best_mse = np.inf
        _best_threshold = max(x)
        for i in sorted(set(x)):
            current_mse = self._weighted_mse(y[x <= i], y[x > i])
            if current_mse < best_mse:
                _best_threshold = i
                best_mse = current_mse
        return _best_threshold, best_mse

    def _best_split(self, X: np.array, y: np.array) -> tuple[int, float]:
        """Find the best split for a node (one feature)"""
        best_feature = np.inf
        best_threshold = np.inf
        best_mse = np.inf
        for i in range(X.shape[1]):
            current_threshold, current_mse = self._split(X, y, i)
            print(current_mse, current_threshold, i)
            if current_mse < best_mse:
                best_mse = current_mse
                best_feature, best_threshold = i, current_threshold
        return best_feature, best_threshold

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0,
                    best_f=None, best_thr=None) -> Node:
        """Split a node and return the resulting left and right child nodes."""

        if depth >= self.max_depth or X.shape[0] <= self.min_samples_split:
            return Node(best_f, best_thr, X.shape[0], int(np.round(y.mean(), 0)), self._mse(y),
                        left=None, right=None)
        best_f, best_thr = self._best_split(X, y)
        X_l, y_l = X[X[:, best_f] <= best_thr], y[X[:, best_f] <= best_thr]
        X_r, y_r = X[X[:, best_f] > best_thr], y[X[:, best_f] > best_thr]
        return Node(best_f, best_thr, X.shape[0], int(np.round(y.mean(), 0)), self._mse(y),
                    self._split_node(X=X_l, y=y_l, depth=depth+1,
                                     best_f=best_f, best_thr=best_thr),
                    self._split_node(X=X_r, y=y_r, depth=depth+1,
                                     best_f=best_f, best_thr=best_thr))
