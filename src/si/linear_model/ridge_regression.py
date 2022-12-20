import numpy as np
import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset
from metrics.mse import mse


class RidgeRegression:
    def __init__(
        self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000
    ):
        """Linear model using the L2 Regularization. Solves the linear regression issue by adapting a Gradient Descent technique. 

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

    def fit(self, dataset: Dataset) -> "RidgeRegression":
        """Estimation of the theta and theta zero for the entry dataset.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            RidgeRegression: Fitted model.
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
        """Estimates the value of predicted Y with the theta parameters.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            np.array: Predicted Y values.
        """
        return np.dot(dataset.x, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """Calculates the mean squared error of the predicted values.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            float: Mean squared error.
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """Computes the cost function between the predicted values and the real ones.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            float: Cost function of the model.
        """
        y_pred = self.predict(dataset)
        return (
            np.sum((y_pred - dataset.y) ** 2)
            + (self.l2_penalty * np.sum(self.theta**2))
        ) / (2 * len(dataset.y))


if __name__ == "__main__":
    # FIXME: Add tests and the evaluation for this methods
    pass
