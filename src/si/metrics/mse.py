import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error of the predicted Y values.

    Args:
        y_true (np.ndarray): Real Y values.
        y_pred (np.ndarray): Predicted Y values.

    Returns:
        np.ndarray: Mean squared error.
    """
    # Function: SUM(Yr, Yp)^2 / 2n
    return np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2)


def mse_derivate(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Derivate mean squared error of the predicted Y values.

    Args:
        y_true (np.ndarray): Real Y values.
        y_pred (np.ndarray): Predicted Y values.

    Returns:
        np.ndarray: Derivate mean squared error.
    """
    # Function: SUM(Yr, Yp)^2 / 2n, order to Yp
    # Derivate: -2 * SUM(Yr, Yp) / 2n
    return -2 * (y_true - y_pred) / (len(y_true) * 2)
