import numpy as np
import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset
from metrics.accuracy import accuracy
from statistics.sigmoid_function import sigmoid_function
import matplotlib.pyplot as plt


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
        self.cost_history = {}

    def fit(self, dataset: Dataset, use_adaptive_alpha: bool = False) -> "LogisticRegression":
        """Estimation of the theta and theta zero for the entry dataset using the sigmoid function Y values.

        Args:
            dataset (Dataset): Dataset input.
            use_adaptive_alpha (bool, optional): Utilizes the different version where the alpha is reduced when reached the cost threshold. Defaults to False.

        Returns:
            LogisticRegression: Fitted model.
        """
        # x_shape == m and y_shape == n
        x_shape, y_shape = dataset.shape()

        # Model parameters
        self.theta = np.zeros(y_shape)
        self.theta_zero = 0
        prev_cost = None
        cost_threshold = 0.0001

        # gradient descent
        for i in range(self.max_iter):
            # Predict Y with the sigmoid function
            y_pred = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

            # Calculate the gradient for the alpha
            # Gradient = alpha * (1/m) * SUM((Predicted Y - Real Y) * X Values)
            gradient = (self.alpha * (1 / x_shape)) * np.dot(
                y_pred - dataset.y, dataset.x
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

            # Create the cost history along the iterations
            curr_cost = self.cost(dataset)
            self.cost_history[i] = curr_cost
            
            # Stop when the cost change is lower than the threshold
            if prev_cost and prev_cost - curr_cost < cost_threshold:
                if not use_adaptive_alpha:
                    break
                self.alpha /= 2

            prev_cost = curr_cost
        return self

    def predict(self, dataset: Dataset) -> np.array:
        """Estimates the value of predicted Y with the sigmoid function.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            np.array: Predicted Y values.
        """
        predictions = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

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
        predictions = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)
        x_shape, _ = dataset.shape()
        add_value = self.l2_penalty / (2 * x_shape) * np.sum(self.theta**2)

        # Fix division from 0 and -inf values (NOT SURE) -> FIXME: np.log(1 - predictions, where=1 - predictions > 0)
        logarithm_value = np.log(1 - predictions)
        cost = (
            -1
            / x_shape
            * np.sum(
                dataset.y * np.log(predictions) + (1 - dataset.y) * logarithm_value
            )
        )
        return cost + add_value

    def cost_plot(self):
        """Design the plot of the cost history along the iterations of the model prediction."""
        plt.plot(self.cost_history.keys(), self.cost_history.values())
        plt.title("Cost History")
        plt.ylabel("Cost")
        plt.xlabel("Iterations")
        plt.show()


if __name__ == "__main__":
    from io_folder.csv_file import read_csv
    from sklearn.preprocessing import StandardScaler
    from model_selection.split import train_test_split

    dataset = read_csv(r"datasets/cpu.csv", features=True, label=6)
    dataset.x = StandardScaler().fit_transform(dataset.x)

    # Split the dataset
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)
    lg = LogisticRegression()
    lg.fit(dataset, use_adaptive_alpha=True)
    print("Score:", lg.score(dataset))
    print("Cost:", lg.cost(dataset))
    lg.cost_plot()
