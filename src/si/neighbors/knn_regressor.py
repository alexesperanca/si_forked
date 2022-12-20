import sys
import numpy as np
from typing import Callable

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from statistics.euclidean_distance import euclidean_distance
from data.dataset import Dataset
from metrics.rmse import rmse


class KNNRegressor:
    def __init__(self, k: int, distance: Callable = euclidean_distance):
        """K-nearest neighbors similar algorithm ti estimate the mean value of the most similar k-examples. More indicated to regression problems.

        Args:
            k (int): K number of examples to analyze.
            distance (Callable, optional): Function to calculate the euclidean distances. Defaults to euclidean_distance.
        """
        self.k = k
        self.distance = distance

    def fit(self, dataset: Dataset) -> "KNNRegressor":
        """Training dataset storage.

        Args:
            dataset (Dataset): Instance of the Dataset class.

        Returns:
            KNNRegressor: Class instance.
        """
        self.training_dataset = dataset
        return self

    def _get_closest_value(self, sample: list) -> float:
        """Get the closest value of the euclidean distances for a given sample. Auxiliary function of self.predict.

        Args:
            sample (list): Sample of a dataset.

        Returns:
            float: Closest value of the sample.
        """
        euclidean_distances = self.distance(sample, self.training_dataset.x)
        # K indexes with the closest distance
        closest_indexes = np.argsort(euclidean_distances)[:self.k]
        y_classes = self.training_dataset.y[closest_indexes]
        return np.mean(y_classes)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """Calculates the estimated classes for the input dataset.

        Args:
            dataset (Dataset): Input dataset.

        Returns:
            np.ndarray: Estimated classes (Y predicted).
        """
        assert self.training_dataset, "No training dataset, please run fit() function with the respective dataset as argument input."
        return np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.x)

    def score(self, dataset: Dataset) -> float:
        """Calculates the accuracy between the real values and the predicted ones.

        Args:
            dataset (Dataset): Input dataset.

        Returns:
            float: Error between the real and the predicted values.
        """
        return rmse(dataset.y, self.predict(dataset))