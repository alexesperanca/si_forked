import sys
import numpy as np

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from statistics.sigmoid_function import sigmoid_function


class Dense:
    def __init__(self, input_size: int, output_size: int):
        """Dense layer is an algorithm where each neuron of a layer is connected to all the neurons in the following layer.

        Args:
            input_size (int): The number of input values the layer will receive.
            output_size (int): The number of outputs the layer will produce.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.x = None

        # Weights of the layer
        self.weights = np.random.randn(input_size, output_size) * 0.01
        # Bias of the layer
        # Represents the shift or offset of the activation function. Allows the model to learn more complex relationships in the data and avoid learning only linear relationships.
        self.bias = np.zeros((1, output_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of the layer using the given input.

        Args:
            x (np.ndarray): Data layer input.

        Returns:
            np.ndarray: Data layer output.
        """
        # Update last inputted value.
        self.x = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward propagation of the layer.

        Args:
            error (np.ndarray): Error of the loss function.
            learning_rate (float): The learning rate value.

        Returns:
            np.ndarray: Error propagation of the previous layer.
        """
        # Update the weight and bias
        self.weights = self.weights - learning_rate * np.dot(self.x.T, error)
        self.bias = self.bias - learning_rate * np.sum(error, axis=0)

        # Error propagations return.
        return np.dot(error, self.weights.T)


class SigmoidActivation:
    def __init__(self):
        """Sigmoid activation layer."""
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of the layer using the given input.

        Args:
            x (np.ndarray): Data layer input.

        Returns:
            np.ndarray: Data layer output.
        """
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Backward propagation of the layer.

        Args:
            x (np.ndarray): Data layer input.
            error (np.ndarray): Error of the loss function.

        Returns:
            np.ndarray: Error propagation of the previous layer.
        """
        sigmoid_derivate = 1 / (1 + np.exp(-x))
        sigmoid_derivate = sigmoid_derivate * (1 - sigmoid_derivate)

        # Get error from previous layer
        return error * sigmoid_derivate


class SoftMaxActivation:
    def __init__(self):
        """SoftMax activation layer."""
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Calculates the occurrence probability of each class.

        Args:
            x (np.ndarray): Data layer input.

        Returns:
            np.ndarray: Data layer output. The occurrence probability of each class.
        """
        exp = np.exp(x)
        # Keep the array dimension
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, x: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Backward propagation of the layer.

        Args:
            x (np.ndarray): Data layer input.
            error (np.ndarray): Error of the loss function.

        Returns:
            np.ndarray: Error propagation of the previous layer.
        """
        return error


class ReLUActivation:
    def __init__(self):
        """ReLU activation layer."""
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of the layer using the given input.

        Args:
            x (np.ndarray): Data layer input.

        Returns:
            np.ndarray: Data layer output. The rectified linear relationship.
        """
        # 0 is the minimum since we can only consider the positive part of the linear function
        return np.maximum(x, 0)

    def backward(self, x: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Backward propagation of the layer.

        Args:
            x (np.ndarray): Data layer input.
            error (np.ndarray): Error of the loss function.

        Returns:
            np.ndarray: Error propagation of the previous layer.
        """
        # Substitute all the values for 1 when higher than 0, otherwise 0
        return error * np.where(x > 0, 1, 0)


class LinearActivation:
    def __init__(self):
        """Linear activation layer.
        Has the characteristic of the returned data being proportional to the given data."""
        pass

    @staticmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of the layer using the given input.

        Args:
            x (np.ndarray): Data layer input.

        Returns:
            np.ndarray: Data layer output. The rectified linear relationship.
        """
        return x
