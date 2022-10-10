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
        return self.x.shape, self.y.shape

    def has_label(self) -> bool:
        """Verifies if we have a dependent variable.

        Returns:
            bool: True if we have. False otherwise.
        """
        return False if self.y is None else True

    def get_classes(self) -> list | None:
        """Get the classes of the dataset (possible values of y)

        Returns:
            list|None: Y classes if possible. Otherwise, return None.
        """
        return None if self.y is None else np.unique(list(self.y))

    def get_mean(self) -> list:
        """Calculate the mean value of the variables.

        Returns:
            list: Numpy array of the mean values of the variables.
        """
        return np.mean(self.x, axis=0)

    def get_variance(self) -> list:
        """Calculate the variance of the variables.

        Returns:
            list: Numpy array of the variance values of the variables.
        """
        return np.var(self.x, axis=0)

    def get_median(self) -> list:
        """Calculate the median of the variables.

        Returns:
            list: Numpy array of the median values of the variables.
        """
        return np.median(self.x, axis=0)

    def get_min(self) -> list:
        """Calculate the minimum value of each variable.

        Returns:
            list: Numpy array of the minimum values of the variables.
        """
        return np.min(self.x, axis=0)

    def get_max(self) -> list:
        """Calculate the maximum value of each variable.

        Returns:
            list: Numpy array of the maximum values of the variables.
        """
        return np.max(self.x, axis=0)

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
        columns = ["Mean", "Variance", "Median", "Minimum", "Maximum"]
        return pd.DataFrame(data, columns)

    def dropna(self) -> pd.DataFrame:
        """Remove all NAs in the dataset.

        Returns:
            pd.DataFrame: Dataset with the NAs removed.
        """
        return pd.DataFrame(self.X).dropna(axis=0).reset_index(drop=True)

    def fillna(self, fill_value: int | str) -> pd.DataFrame:
        """Replaces all NAs in the dataset.

        Returns:
            pd.DataFrame: Dataset with the NAs replaced.
        """
        return pd.DataFrame(self.X).fillna(fill_value)
