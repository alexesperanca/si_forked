import sys
from typing_extensions import Self
import numpy as np
import logging

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from statistics.euclidean_distance import euclidean_distance
from typing import Callable


class KMeans:
    def __init__(
        self, k: int, max_iter: int, distance: Callable = euclidean_distance
    ) -> None:
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        self.centroides = None
        self.labels = []

    def fit(self, data: list):
        # FIXME: This if annoying me...
        self.centroides = [
            list(np.random.permutation(len(data[0]))) for _ in range(self.k)
        ]
        iterations = 0
        for sample in data:
            distances = self.distance(sample, self.centroides)
            min_dist_idx = distances.index(min(distances))
            # Centroid with the lowest distance
            min_dist_cent = self.centroides[min_dist_idx]
            self.labels.append(np.mean(min_dist_cent))
        print(self.labels)
        print(len(self.labels))
        print(len(data))

    def transform(self, data: list):
        """Obtains all the distances of the samples in the data and the centroids.

        Args:
            data (list): Initial data input.
        """
        return [self.distance(sample, self.centroides) for sample in data]

    def predict(self, data: list):
        """Obtains the labels of the data.

        Args:
            data (list): Initial data input.
        """
        distances = self.transform(data)
        labels = []
        for idx, _ in enumerate(data):
            min_dist_idx = distances[idx].index(min(distances[idx]))
            # Centroid with the lowest distance
            min_dist_cent = self.centroides[min_dist_idx]
            labels.append(min_dist_cent)
        self.labels = labels


if __name__ == "__main__":
    import pandas as pd
    from sklearn import preprocessing

    iris = pd.read_csv(
        r"C:\Users\alexandre.esperanca\Desktop\Learning\si_forked\datasets\iris.csv",
        sep=",",
        index_col=4,
    )
    iris_scale = preprocessing.scale(iris.iloc[:, :4])
    test = KMeans(k=3, max_iter=10)
    test.fit(data=iris_scale)
