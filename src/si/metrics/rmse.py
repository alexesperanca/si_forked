import numpy as np


def rmse(y_true: list, y_pred: list) -> float:
    """Root Mean Square Error calculation.

    Args:
        y_true (list): Real Y axis values.
        y_pred (list): Predicted Y axis values.

    Returns:
        float: RMSE error.
    """
    return np.sqrt(np.sum(np.subtract(y_true, y_pred) ** 2) / len(y_true))
