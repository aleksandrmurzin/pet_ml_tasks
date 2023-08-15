from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    mean_squared_error = np.mean(np.square(y - np.mean(y)))
    return mean_squared_error


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    weighted_mean_squared_error = (
        (mse(y_left) * len(y_left) + mse(y_right) * len(y_right))
        / (len(y_left) + len(y_right)))
    return weighted_mean_squared_error


def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split for a node (one feature)"""
    x = X[:, feature]
    best_mse = np.inf
    for i in sorted(set(x))[:-1]:
        current_mse = weighted_mse(x[x<=i], x[x>i])
        if current_mse < best_mse:
            best_threshold = i
            best_mse = current_mse
    return best_threshold


def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Find the best split for a node (one feature)"""
    best_feature = np.inf
    best_threshold = np.inf
    best_mse = np.inf
    for i in range(X.shape[1]):
        current_threshold = split(X, y, i)
        current_mse = weighted_mse(X[:, i][X[:, i]<=current_threshold],
                                   X[:, i][X[:, i]>current_threshold])
        if current_mse < best_mse:
            best_mse = current_mse
            best_feature, best_threshold = i, current_threshold
    return best_feature, best_threshold



class Node:
    """Decision tree node."""
    def __init__(self, feature = None, threshold = None, n_samples = None,
                 value = None, mse = None, left: Node = None, right: Node = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.n_samples = n_samples
        self.value = value
        self.mse = mse
        self.left = left
        self.right = right