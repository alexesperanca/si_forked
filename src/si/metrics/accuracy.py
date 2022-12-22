import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the accuracy between 2 lists.

    Args:
        y_true ( np.ndarray): Values of the variable.
        y_pred ( np.ndarray): Predicted values.

    Returns:
        float: Accuracy result of the prediction
    """
    correct_values = len([value for value in y_pred if value in y_true])
    return (correct_values / len(y_true)) * 100
