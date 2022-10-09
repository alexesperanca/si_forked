import sys
import numpy as np

DATASET_CLASS_PATH = "src/si"
sys.path.insert(0, DATASET_CLASS_PATH)

from data.dataset import Dataset


class VarianceThreshold:
    def __init__(self, threshold: int) -> None:
        """Store the threshold input value.

        Args:
            threshold (int): Threshold value, considered the cutting value of the estimated parameters.
        """
        assert threshold >= 0, "Threshold should be a non-negative value."
        self.threshold = threshold

    def fit(self, dataset: Dataset):
        """Calculates the variance of each feature.

        Args:
            dataset (Dataset): Dataset input.
        """
        self.variance = dataset.get_variance()
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """Select all the features with variance above the threshold.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            Dataset: Dataset with new X axis.
        """
        condition = self.variance > self.threshold
        x = dataset.x[:, condition]
        features = np.array(dataset.features)[condition]
        return Dataset(x, dataset.y, features, None)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Filters the input dataset with the methods fit and transform.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            Dataset: Dataset inputted after filtering.
        """
        self.fit(dataset)
        return self.transform(dataset)
