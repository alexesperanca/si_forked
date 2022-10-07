import numpy as np


class VarianceThreshold:
    def __init__(self, threshold: int) -> None:
        """TODO:

        Args:
            threshold (int): Threshold value, considered the cutting value of the estimated parameters.
        """
        assert threshold >= 0, "Threshold should be a non-negative value."
        self.threshold = threshold

    def fit(self, dataset):
      """_summary_

      Args:
          dataset (TODO: Confirm this): _description_
      """
        x_axis = dataset.X
        self.variance = np.var(x_axis, axis=0)