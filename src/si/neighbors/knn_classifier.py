import sys
import numpy as np
from typing import Callable

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from statistics.euclidean_distance import euclidean_distance
from data.dataset import Dataset


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

    def predict(self, dataset: np.array) -> list:
        samples = dataset.x
        distances = []
        for sample in samples:
          euclidean_distances = self.distance(sample, self.training_dataset.x)
          closest_index = np.argsort(euclidean_distances)
          print(closest_index)

    def score(self):
        pass


if __name__ == "__main__":
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]]
    )  # 4r, 4c

    y = np.array([10, 20, 30, 10])  # 4r, 1c
    
    x1 = np.array(
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28]]
    )  # 4r, 4c

    y1 = np.array([40, 30, 70, 40])  # 4r, 1c
    dataset1 = Dataset(x, y)
    dataset2 = Dataset(x1, y1)
    
    test = KNNClassifier(3)
    test.fit(dataset1)
    test.predict(dataset2)
