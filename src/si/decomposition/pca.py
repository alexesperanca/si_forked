import numpy as np


class PCA:
    def __init__(self, n_components: int):
        """Storage of the input values.

        Args:
            n_components (int): Number of components.
        """
        self.n_components = n_components
        self.mean = None
        self.svd = None
        self.principal_components = None
        self.explained_variance = None
        self.data_centered = None

    def fit(self, dataset: object) -> object:
        """Centers the data provided, calculates the principal components, and the explained variance.

        Args:
            dataset (object): Input dataset.

        Returns:
            object: Self object.
        """
        self.mean = np.mean(dataset.x, axis=0)
        self.data_centered = np.subtract(dataset.x, self.mean)
        u, s, vt = np.linalg.svd(self.data_centered, full_matrices=False)
        self.svd = u * s * vt

        # Get the principal components
        self.principal_components = vt[: self.n_components]

        # Variance
        n = len(dataset.x[0])
        ev = s**2 / (n - 1)
        self.explained_variance = ev[: self.n_components]
        return self

    def transform(self, dataset: object) -> list:
        """Calculates the reduced X axis from the dataset.

        Args:
            dataset (object): Input dataset.

        Returns:
            list: Reduced X axis.
        """
        # If the function fit was not called previously 
        if self.mean is None:
            self.fit(dataset)
        
        # Transpose matrix and reduce the centered data
        v = self.principal_components.T
        x_reduced = np.dot(self.data_centered, v)
        return x_reduced
