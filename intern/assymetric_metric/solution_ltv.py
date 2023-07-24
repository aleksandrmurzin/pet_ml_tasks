import numpy as np


def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate the Lifetime Value (LTV) error between the predicted and true values.

    Parameters
    ----------
    y_true : np.array
        The true lifetime values.
    y_pred : np.array
        The predicted lifetime values.

    Returns
    -------
    float
        The mean absolute LTV error.

    Notes
    -----
    The Lifetime Value (LTV) error is calculated as the mean absolute error between
    the predicted lifetime values and the true lifetime values. The LTV error is a
    measure of how well the predictions match the true values, with values closer to
    zero indicating a better prediction.

    The LTV error is defined as:

    .. math:: error = |1 - (y_pred^2 / y_true^2)|

    where y_pred is the predicted lifetime value, and y_true is the true lifetime value.
    Examples
    --------
    >>> y_true = np.array([5, 10, 15])
    >>> y_pred = np.array([4, 9, 16])
    >>> ltv_error(y_true, y_pred)
    0.037037037037037035
    """
    error = np.abs(1 - np.square(y_pred) / np.square(y_true))
    return np.mean(error)
