import numpy as np
import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset
from metrics.accuracy import accuracy
from statistics.sigmoid_function import sigmoid_function


class LogisticRegression:
    def __init__(
        self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000
    ):
        """Logistic model using the L2 Regularization. Solves the linear regression issue by adapting a Gradient Descent technique.

        Args:
            l2_penalty (float, optional): L2 regularization. Defaults to 1.
            alpha (float, optional): Learning rate. Defaults to 0.001.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        """
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = None
        self.theta_zero = None

    def fit(self, dataset: Dataset) -> "LogisticRegression":
        """Estimation of the theta and theta zero for the entry dataset using the sigmoid function Y values.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            LogisticRegression: Fitted model.
        """
        # x_shape == m and y_shape == n
        x_shape, y_shape = dataset.shape()

        # Model parameters
        self.theta = np.zeros(y_shape)
        self.theta_zero = 0

        # gradient descent
        for _ in range(self.max_iter):
            # Predict Y
            # FIXME: Doubts in this equation
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # apply sigmoid function
            y_pred = sigmoid_function(y_pred)

            # Calculate the gradient for the alpha
            # Gradient = alpha * (1/m) * SUM((Predicted Y - Real Y) * X Values)
            gradient = (self.alpha * (1 / x_shape)) * np.dot(
                y_pred - dataset.y, dataset.X
            )

            # Regularization term
            # theta * (1 - alpha * (l2/m))
            # FIXME: Missing the value 1?
            penalization_term = self.theta * self.alpha * (self.l2_penalty / x_shape)

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / x_shape)) * np.sum(
                y_pred - dataset.y
            )
        return self

    def predict(self, dataset: Dataset) -> np.array:
        """Estimates the value of predicted Y with the sigmoid function.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            np.array: Predicted Y values.
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # Convert the predictions to 0 or 1
        mask = predictions >= 0.5
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

    def score(self, dataset: Dataset) -> float:
        """Calculates the accuracy of the predicted values.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            float: Predict accuracy value.
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """Computes the cost function between the predicted values and the real ones.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            float: Cost function of the model.
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - (
            (1 - dataset.y) * np.log(1 - predictions)
        )
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (
            self.l2_penalty * np.sum(self.theta**2) / (2 * dataset.shape()[0])
        )
        return cost


if __name__ == "__main__":
    # FIXME: Add tests and the evaluation for this methods
    pass