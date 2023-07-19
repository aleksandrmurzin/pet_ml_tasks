import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates the Symmetric Mean Absolute Percentage 
    Error (SMAPE) between the true and predicted values.

    Parameters:
        y_true (np.array): The true values.
        y_pred (np.array): The predicted values.

    Returns:
        float: The SMAPE score.

    """
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)

    divide = np.divide(numerator.astype(np.float64), denominator.astype(np.float64),
                       casting='unsafe',
                       out=np.zeros_like(numerator, dtype=np.float64), where=denominator!=0)

    return np.mean(divide, dtype=np.float64)
