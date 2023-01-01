import sys
import numpy as np

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from neural_network.layer import *
from data.dataset import Dataset
from metrics.accuracy import accuracy
from metrics.mse import mse
from metrics.mse import mse_derivate


class NN:
    """Neural network possesses several layers to compose a model of values prediction. It runs as the follow:
    1. Fit the neural network.
        This function runs the forward propagation through all the layers to obtain a predicted error value.
        With this, we run the backward propagation to update each layer weight and bias to be more accurate, reducing the error.
        We may verify the learning of all the layers by the maintenance of an history with the cost on each epoch.
        This function runs the number of epochs passed initially. Important value to pass carefully.
    2. Predict the output layer.
        This makes the last forward propagation through each layer after the fit trained the model.
        This last forward propagation is running with all the weights and bias updated, with the error reduced the maximum it could."""

    def __init__(
        self,
        layers: list,
        epochs: int = 1000,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> None:
        self.layers = layers
        # Number of times the training data will be analyzed to reduce in each iteration the predicted error
        # Can be a tricky value, since a low value may result in a bad training, and a high value may overfit the model (become to specific)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.history = {}

    def fit(self, dataset: Dataset) -> "NN":
        """Model training.We propagate forward and backwards the number of epochs passed initially.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            NN: Fitted model.
        """
        x, y = (dataset.x, dataset.y)
        # Train the model n (epochs) times
        for epoch in range(1, self.epochs + 1):
            # Forward propagation of the input layer data
            for layer in self.layers:
                x = layer.forward(x)

            # calculate the associated error with the predicted values
            error = mse_derivate(y, x)

            # Backward propagation of the input layer data
            for layer in self.layers[::-1]:
                error = layer.backward(error, self.learning_rate)

            cost = mse(y, x)
            self.history[epoch] = cost

            if self.verbose:
                print(f"Epoch n{epoch} cost value: {cost}")

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """Iterate throw all layers and predict the final output values of the neural network.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            np.ndarray: Predicted output values.
        """
        x = dataset.x
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def score(self, dataset: Dataset) -> float:
        """Computes the accuracy between the predicted values and the real ones.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            float: Accuracy value of the model.
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """Computes the mean squared error between the predicted values and the real ones.

        Args:
            dataset (Dataset): Dataset input.

        Returns:
            float: Mean squared error value of the model.
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)


if __name__ == "__main__":
    # Construct the layers
    layer1 = Dense(32, 32)
    layer2 = Dense(32, 16)
    layer3 = Dense(16, 1)

    # Evaluation 10.3
    layer1_sg_activation = SigmoidActivation()
    layer2_sg_activation = SigmoidActivation()
    layer3_sg_activation = SigmoidActivation()

    nn1 = NN(
        layers=[
            layer1,
            layer1_sg_activation,
            layer2,
            layer2_sg_activation,
            layer3,
            layer3_sg_activation,
        ]
    )

    # Evaluation 10.4
    layer1_sg_activation = SigmoidActivation()
    layer2_sg_activation = SigmoidActivation()
    layer3_sm_activation = SoftMaxActivation()

    nn2 = NN(
        layers=[
            layer1,
            layer1_sg_activation,
            layer2,
            layer2_sg_activation,
            layer3,
            layer3_sm_activation,
        ]
    )

    # Evaluation 10.5
    layer1_relu_activation = ReLUActivation()
    layer2_relu_activation = ReLUActivation()
    layer3_linear_activation = LinearActivation()

    nn3 = NN(
        layers=[
            layer1,
            layer1_relu_activation,
            layer2,
            layer2_relu_activation,
            layer3,
            layer3_linear_activation,
        ]
    )

    # Construct the Datasets
    dataset = Dataset.from_random(64, 32)

    y_test1 = np.random.randint(0, 2, 64)  # Binary, so only 0 or 1
    dataset_test1 = Dataset(dataset.x, y_test1)

    y_test2 = np.random.randint(0, 3, 64)  # 3 classes
    dataset_test2 = Dataset(dataset.x, y_test2)

    y_test3 = (
        np.random.rand(64) * 100
    )  # Regression problem -> Output values are continuous
    dataset_test3 = Dataset(dataset.x, y_test3)

    # Predict the models
    nn1.fit(dataset)
    nn1.predict(dataset_test1)
