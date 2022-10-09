import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from statistics.f_classification import f_classification
from typing import Callable

import numpy as np
from data.dataset import Dataset


class SelectKBest:
    def __init__(self, k: int, score_func: Callable = f_classification) -> None:
        """Storage of the input values.

        Args:
            score_func (Callable, optional): Calls the inputted function. Defaults to f_classification.
            k (int): Number of features to select.
        """
        assert k >= 0, "K should be a non-negative value."
        self.score_func = score_func
        self.k = k
        self.f = None
        self.p = None

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
        indices = np.argsort(self.f)[-self.k :][::-1]
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
