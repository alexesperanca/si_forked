import numpy as np
import pandas as pd


class Dataset:
    def __init__(
        self,
        x: list,
        y: list = None,
        features: list = None,
        label: str = None,
    ) -> None:
        """Storage of the input values.

        Args:
            x (list): Matrix and table of features.
            y (list, optional): Variable dependent vector. Defaults to None.
            features (list, optional): Features name. Defaults to None.
            label (str, optional): Dependent variable vectors name. Defaults to None.
        """
        self.x = x
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> tuple:
        """Obtain the shape of the dataset.

        Returns:
            tuple: Axis shape.
        """
        return self.x.shape

    def has_label(self) -> bool:
        """Verifies if we have a dependent variable.

        Returns:
            bool: True if we have. False otherwise.
        """
        return False if self.y is None else True

    def get_classes(self) -> np.ndarray:
        """Get the classes of the dataset (possible values of y)

        Returns:
            np.ndarray|None: Y classes if possible. Otherwise, return None.
        """
        return None if self.y is None else np.unique(list(self.y))

    def get_mean(self) -> np.ndarray:
        """Calculate the mean value of the variables.

        Returns:
            np.ndarray: Numpy array of the mean values of the variables.
        """
        return np.ndarray(np.mean(self.x, axis=0))

    def get_variance(self) -> np.ndarray:
        """Calculate the variance of the variables.

        Returns:
            np.ndarray: Numpy array of the variance values of the variables.
        """
        return np.ndarray(np.var(self.x, axis=0))

    def get_median(self) -> np.ndarray:
        """Calculate the median of the variables.

        Returns:
            np.ndarray: Numpy array of the median values of the variables.
        """
        return np.ndarray(np.median(self.x, axis=0))

    def get_min(self) -> np.ndarray:
        """Calculate the minimum value of each variable.

        Returns:
            np.ndarray: Numpy array of the minimum values of the variables.
        """
        return np.ndarray(np.min(self.x, axis=0))

    def get_max(self) -> np.ndarray:
        """Calculate the maximum value of each variable.

        Returns:
            np.ndarray: Numpy array of the maximum values of the variables.
        """
        return np.ndarray(np.max(self.x, axis=0))

    def summary(self) -> pd.DataFrame:
        """Construction of a Dataframe that resumes all the metrics.

        Returns:
            pd.DataFrame: Metrics of the dataset.
        """
        data = [
            self.get_mean(),
            self.get_variance(),
            self.get_median(),
            self.get_min(),
            self.get_max(),
        ]
        return pd.DataFrame(
            list(zip(*data)),
            columns=["Mean", "Variance", "Median", "Minimum", "Maximum"],
        )

    def dropna(self) -> pd.DataFrame:
        """Remove all NAs in the dataset.

        Returns:
            pd.DataFrame: Dataset with the NAs removed.
        """
        return pd.DataFrame(self.x).dropna(axis=0).reset_index(drop=True)

    def fillna(self, fill_value: int) -> pd.DataFrame:
        """Replaces all NAs in the dataset.

        Returns:
            pd.DataFrame: Dataset with the NAs replaced.
        """
        return pd.DataFrame(self.x).fillna(fill_value)

    def from_random(
        n_samples: int,
        n_features: int,
        n_classes: int = 2,
        features: list = None,
        label: str = None,
    ) -> "Dataset":
        """Creation of a Dataset Object from random data input.

        Args:
            n_samples (int): Number of samples.
            n_features (int): Number of features.
            n_classes (int, optional): Number of classes. Defaults to 2.
            features (list, optional): List of features. Defaults to None.
            label (str, optional): Label name. Defaults to None.

        Returns:
            Dataset: Dataset output.
        """
        x = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return Dataset(x, y, features=features, label=label)
