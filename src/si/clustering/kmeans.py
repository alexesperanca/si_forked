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
        self.centroids = None
        self.labels = []

    def fit(self, data: list):
        """Gets the labels without difference between the samples

        Args:
            data (list): Input data samples.
        """
        dataframe = pd.DataFrame(data)

        # Init centroids
        self.centroids = [
            list(np.random.permutation(len(data[0]))) for _ in range(self.k)
        ]
        niter = 0
        while niter < self.max_iter:
            previous_labels = self.labels
            niter += 1
            # Get the labels
            self.predict(data)
            # End if we have no change in the labels
            if previous_labels == self.labels:
                break
            
            dataframe["centroid"] = self.labels
            self.centroids = (
                dataframe.groupby("centroid")
                .agg("mean")
                .reset_index(drop=True)
                .values.tolist()
            )

    def transform(self, data: list) -> list:
        """Obtains all the distances of the samples in the data and the centroids.

        Args:
            data (list): Initial data input.

        Returns:
            list: Euclidean distances between each sample and the centroids
        """
        return [self.distance(sample, self.centroids) for sample in data]

    def predict(self, data: list):
        """Obtains the labels of the data.

        Args:
            data (list): Input data samples.
        """
        distances = self.transform(data)
        closest_centroids = []
        for sample in data:
            distances = self.distance(sample, self.centroids)
            min_dist_idx = distances.index(min(distances))
            # Centroid with the lowest distance
            closest_centroids.append(min_dist_idx)
        self.labels = closest_centroids


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
    print(test.labels)
