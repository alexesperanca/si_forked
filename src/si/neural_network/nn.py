import sys
import numpy as np

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset


class NN:
  def __init__(self, layers: list = []) -> None:
    self.layers = layers
    
  def forward(self, dataset: Dataset):
    # FIXME: Finish this
    pass