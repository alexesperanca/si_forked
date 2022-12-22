import numpy as np


def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """Calculate the sigmoid function of the inputted values.

    Args:
        X (np.ndarray): Input values.

    Returns:
        np.ndarray: Sigmoid function value.
    """
    return 1 / (1 + np.exp(-X))
