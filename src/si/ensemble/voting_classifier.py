import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

import numpy as np
from data.dataset import Dataset
from metrics.accuracy import accuracy


class VotingClassifier:
    def __init__(self, models: list):
        """Ensemble classifier to predict class labels through a majority vote of models.

        Args:
            models (list): Models to use in the ensemble.
        """
        self.models = models

    def fit(self, dataset: Dataset) -> "VotingClassifier":
        """Fit each model passed to the respective dataset

        Args:
            dataset (Dataset): Data input.

        Returns:
            VotingClassifier: Fitted model.
        """
        # Train each model
        for model in self.models:
            model.fit(dataset)
        return self

    def _get_most_recurrent_prediction(self, predictions: np.ndarray) -> int:
        """Returns the most recurrent prediction of all the models inputted.

        Args:
            predictions (np.ndarray): Model predictions.

        Returns:
            int: Most recurrent value predicted.
        """
        labels, counts = np.unique(predictions, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.ndarray:
        """Predict the class value for the samples in the dataset.

        Args:
            dataset (Dataset): Data input.

        Returns:
            np.ndarray: Predicted class value (in quantitative value).
        """
        predictions = np.array(
            [model.predict(dataset) for model in self.models]
        ).transpose()
        # Apply a function along the 1 Dimensional axis to obtain the most recurrent models' prediction.
        return np.apply_along_axis(
            self._get_most_recurrent_prediction, axis=1, arr=predictions
        )

    def score(self, dataset: Dataset) -> float:
        """Calculates the mean accuracy between the data input and the forecast prediction.

        Args:
            dataset (Dataset): Data input.

        Returns:
            float: Mean accuracy.
        """
        return accuracy(dataset.y, self.predict(dataset))


if __name__ == "__main__":
    from data.dataset import Dataset
    from io_folder.csv_file import read_csv
    from sklearn.preprocessing import StandardScaler
    from model_selection.split import train_test_split
    from neighbors.knn_classifier import KNNClassifier
    from linear_model.logistic_regression import LogisticRegression

    dataset = read_csv(r"datasets\breast-bin.csv", label=-1)
    dataset.X = StandardScaler().fit_transform(dataset.x)

    # Split the dataset
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)

    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize the Voting classifier
    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    # compute the score
    score = voting.score(dataset_test)
    print(f"Score: {score}")