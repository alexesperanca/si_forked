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

    # Weights of the layer
    self.weights = np.random.randn(input_size, output_size) * 0.01
    # Bias of the layer
    self.bias = np.zeros((1, output_size))
  
  def forward(self, x: np.ndarray) -> np.ndarray:
    """Forward propagation of the layer using the given input.

    Args:
        x (np.ndarray): Layer input.

    Returns:
        np.ndarray: Layer output.
    """
    return np.dot(x, self.weights) + self.bias
  
  def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
    """Backward propagation of the layer.

    Args:
        error (np.ndarray): FIXME: Error from the further layers in the neural network.
        learning_rate (float): _description_

    Returns:
        np.ndarray: _description_
    """
    return error
  
  
class SigmoidActivate:
  def __init__(self):
    """Sigmoid activation layer.
    """
    self.x = None
  
  def forward(self, x: np.ndarray) -> np.ndarray:
    """Forward propagation of the layer using the given input.

    Args:
        x (np.ndarray): Layer input.

    Returns:
        np.ndarray: Layer output.
    """
    return 1 / (1 + np.exp(-x))

  def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
    """Backward propagation of the layer.

    Args:
        error (np.ndarray): FIXME: Error from the further layers in the neural network.
        learning_rate (float): _description_

    Returns:
        np.ndarray: _description_
    """
    # sigmoid_derivate = 1 / (1 + np.exp(-self.x))
    # sigmoid_derivate = sigmoid_derivate * (1 - sigmoid_derivate)
    
    # Get error from previous layer
    # error_to_propagate = error * sigmoid_derivate
    return error
  

class SoftMaxActivation:
  def __init__(self):
    pass
  

class ReLUActivation:
  def __init__(self):
    pass