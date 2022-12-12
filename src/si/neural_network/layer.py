import sys
import numpy as np

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from statistics.sigmoid_function import sigmoid_function


class Dense:
  def __init__(self, input_size: int, output_size: int):
    self.input_size = input_size
    self.output_size = output_size
  
class SigmoidActivate:
  def __init__(self):
    self.X = None
  
  def forward(self, x: np.ndarray) -> np.ndarray:
    pass

  def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
    sigmoid_derivate = 1 / (1 + np.exp(-self.X))
    sigmoid_derivate = sigmoid_derivate * (1 - sigmoid_derivate)
    
    # Get error from previous layer
    error_to_propagate = error * sigmoid_derivate
    