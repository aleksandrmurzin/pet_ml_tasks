import numpy as np


def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate the turnover error between true and predicted values.

    This function calculates the turnover error between two arrays of values,
    representing the true values and the predicted values respectively. The
    turnover error is a measure of the discrepancy between the true and predicted
    values, taking into account cases where the true value is zero.

    Args:
        y_true (np.array): An array of true values.
        y_pred (np.array): An array of predicted values.

    Returns:
        float: The turnover error between the true and predicted values.
    """
    numerator = y_true - y_pred
    error = np.divide(numerator, y_pred, casting="unsafe",
                      out=np.zeros_like(numerator, dtype=np.float64),
                      where=y_true != 0)
    return np.sum(np.square(error))
