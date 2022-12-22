import sys
import numpy as np
import pandas as pd

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset
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

    def fit(self, dataset: np.ndarray):
        """Gets the labels without difference between the samples

        Args:
            data (np.ndarray): Input data samples.
        """
        
        df = pd.DataFrame(dataset.x)
        # Init centroids
        self.centroids = [
            list(np.random.permutation(dataset.shape()[0])) for _ in range(self.k)
        ]
        niter = 0
        while niter < self.max_iter:
            previous_labels = self.labels
            niter += 1
            # Get the labels
            self.predict(dataset.x)
            # End if we have no change in the labels
            if previous_labels == self.labels:
                break

            df["centroid"] = self.labels
            self.centroids = (
                df.groupby("centroid")
                .agg("mean")
                .reset_index(drop=True)
                .values.tolist()
            )

    def transform(self, dataset: Dataset) -> list:
        """Obtains all the distances of the samples in the dataset and the centroids.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            list: Euclidean distances between each sample and the centroids
        """
        print(self.centroids)
        return [self.distance(sample, self.centroids) for sample in dataset]

    def predict(self, dataset: Dataset):
        """Obtains the labels of the dataset.

        Args:
            dataset (Dataset): Dataset input.
        """
        distances = self.transform(dataset)
        closest_centroids = []
        for sample in dataset:
            distances = self.distance(sample, self.centroids)
            min_dist_idx = distances.index(min(distances))
            # Centroid with the lowest distance
            closest_centroids.append(min_dist_idx)
        return closest_centroids


if __name__ == "__main__":
    from io_folder.csv_file import read_csv

    iris = read_csv(
        r"C:\Users\alexandre.esperanca\Desktop\Learning\si_forked\datasets\iris.csv",
        label=4,
    )
    kmeans = KMeans(k=3, max_iter=10)
    kmeans.fit(iris)
    print(kmeans.predict(iris))
