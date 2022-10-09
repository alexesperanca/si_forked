import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from statistics.f_classification import f_classification
from typing import Callable

import numpy as np
from data.dataset import Dataset


class SelectPercentile:
    def __init__(
        self, percentile: int, score_func: Callable = f_classification
    ) -> None:
        """Storage of the input values.

        Args:
            percentile (int): Percentile of features to select.
            score_func (Callable, optional): Calls the inputted function. Defaults to f_classification.
        """
        assert (
            percentile > 0 or percentile < 1
        ), "Percentile should be a value between 0 and 1."
        self.score_func = score_func
        self.percentile = percentile

    def fit(self, dataset: Dataset):
        """Obtains the F-scores and p-values of the variables in the dataset.

        Args:
            dataset (Dataset): Dataset input.
        """
        self.f, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """Select all the features with variance above the threshold.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            Dataset: Dataset with new X axis.
        """
        size = len(dataset.features)
        percentile = int(size * self.percentile)
        indices = np.argsort(self.f)[-percentile:]
        features = np.array(dataset.features)[indices]
        return Dataset(dataset.x[:, indices], dataset.y, features, dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Filters the input dataset with the methods fit and transform.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            Dataset: Dataset inputted after filtering.
        """
        self.fit(dataset)
        return self.transform(dataset)
