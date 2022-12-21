import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

import numpy as np
from data.dataset import Dataset
from metrics.accuracy import accuracy


class StackingClassifier:
    def __init__(self, models: list, final_model: object):
        """Ensemble classifier to generate predictions through a majority vote of models. Those predictions are then used to train a final model inputted.

        Args:
            models (list): Models to use in the ensemble.
            final_model (object): Final model object.
        """
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> "StackingClassifier":
        """Train each model and predict the result. The predictions will be incorporated to form the train data for the final model.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            StackingClassifier: Fitted model.
        """
        predicted_data = dataset.x

        # Train each model
        for model in self.models:
            model.fit(dataset)
            predicted_data = np.c_[predicted_data, model.predict(dataset)]

        # Train final model
        self.final_model.fit(
            Dataset(predicted_data, dataset.y, dataset.features, dataset.label)
        )
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """Predicts the data of the final model through training of data joined from the other models

        Args:
            dataset (Dataset): Dataset input

        Returns:
            np.ndarray: Final output predictions.
        """
        predicted_data = dataset.x

        # Train each model
        for model in self.models:
            predicted_data = np.c_[predicted_data, model.predict(dataset)]

        return self.final_model.predict(
            Dataset(predicted_data, dataset.y, dataset.features, dataset.label)
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

    # initialize the final_model
    final_model = KNNClassifier(k=3)

    model = StackingClassifier([knn, lg], final_model)

    model.fit(dataset_train)

    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")
