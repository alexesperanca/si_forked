import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from statistics.euclidean_distance import euclidean_distance
from data.dataset import Dataset
from typing import Callable


class KNNClassifier:
    def __init__(self, k: int, distance: Callable = euclidean_distance):
        self.k = k
        self.distance = distance

    def fit(self, dataset: object) -> object:
        """Training dataset storage.

        Args:
            dataset (object): Instance of the Dataset class.

        Returns:
            object: Class instance.
        """
        self.training_dataset = dataset
        return self

    def predict(self, test_dataset: object) -> list:
        distances = self.distance(test_dataset.astype(float), self.dataset.x.astype(float))
        print(distances)

    def score(self):
        pass
      
if __name__ == "__main__":
    
