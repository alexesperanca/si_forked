import numpy as np


class Dataset:
    def __init__(self, x=None, y=None, features: list = None, label: str = None):
        self.x = x
        self.y = y

    def shape(self):
        return self.X.shape[0]

    def has_label(self):
        return self.y

    def get_classes(self):
        pass

    def get_mean(self):
        pass

    def get_variance(self):
        pass

    def get_median(self):
        pass

    def get_min(self):
        pass

    def get_max(self):
        pass

    def summary(self):
        pass
