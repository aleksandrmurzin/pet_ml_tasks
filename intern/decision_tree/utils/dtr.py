from __future__ import annotations
from dataclasses import dataclass

import numpy as np


class Node:
    """Decision tree node."""
    def __init__(self, feature = None, threshold = None, n_samples = None,
                 values = None, mse = None, left: Node = None, right: Node = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.n_samples = n_samples
        self.values = values
        self.mse = mse
        self.left = left
        self.right = right


@dataclass
class DesicisionTreeRegressor:
    """Desision tree regressor"""

    max_depth: int
    max_samples_split: int = 2

    def fit(self, X: np.array, y: np.array) -> DesicisionTreeRegressor:
        self.n_features = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return None

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

    def _best_split(self, X: np.array, y: np.array) -> tuple[int, float]:
        """Find the best split for a node (one feature)"""

        def _split(X: np.array, y: np.array, feature: int) -> float:
            """Find the best split for a node (one feature)"""
            x = X[:, feature]
            best_mse = np.inf
            for i in sorted(set(x))[:-1]:
                current_mse = self._weighted_mse(x[x <= i], x[x > i])
                if current_mse < best_mse:
                    best_threshold = i
                    best_mse = current_mse
            return best_threshold
   
        best_feature = np.inf
        best_threshold = np.inf
        best_mse = np.inf
        for i in range(X.shape[1]):
            current_threshold = _split(X, y, i)
            current_mse = self._weighted_mse(
                X[:, i][X[:, i] <= current_threshold],
                X[:, i][X[:, i] > current_threshold])
            if current_mse < best_mse:  
                best_mse = current_mse
                best_feature, best_threshold = i, current_threshold
        return best_feature, best_threshold
    
    def _split_node(self, feature = None, threshold = None, n_samples = None,
                 values = None, mse = None, left: Node = None, 
                 right: Node = None) -> None:
        
        left, right = self._best_split(), self._best_split()
        node = Node(feature=feature, threshold=threshold, n_samples=n_samples,
                    values=values, mse=mse, left=None, right=None)
        return node