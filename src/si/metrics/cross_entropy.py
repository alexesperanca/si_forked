import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the cross entropy loss function.

    Args:
        y_true (np.ndarray): Real Y values.
        y_pred (np.ndarray): Predicted Y values.

    Returns:
        float: Cross entropy loss value.
    """
    # Function: - SUM(Yi * ln(Yp)) / n
    return -np.sum(y_true) * np.log(y_pred) / len(y_true)


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the derivate cross entropy loss function.

    Args:
        y_true (np.ndarray): Real Y values.
        y_pred (np.ndarray): Predicted Y values.

    Returns:
        float: derivate cross entropy loss value.
    """
    return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)
