import numpy as np


def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """Calculate the sigmoid function of the inputted values.

    Args:
        x (np.ndarray): Input values.

    Returns:
        np.ndarray: Sigmoid function value.
    """
    x = x.astype(np.float)
    return 1 / (1 + np.exp(-x))
