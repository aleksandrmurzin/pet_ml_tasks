import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """
    """
    numerator = y_true - y_pred
    error = np.divide(numerator, y_pred, casting="unsafe", 
                      out=np.zeros_like(numerator, dtype=np.float64),
                      where=y_true!=0)
    return np.sum(np.square(error))
