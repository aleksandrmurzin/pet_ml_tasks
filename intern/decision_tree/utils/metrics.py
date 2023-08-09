import numpy as np

def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    mean_squared_error = np.mean(np.square(y - np.mean(y)))
    return mean_squared_error


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    weighted_mean_squared_error = ((mse(y_left) * len(y_left) + mse(y_right) * len(y_right))
                                   / (len(y_left) + len(y_right)))
    return weighted_mean_squared_error