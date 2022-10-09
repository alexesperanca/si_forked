import sys

DATASET_CLASS_PATH = "src/si"
sys.path.insert(0, DATASET_CLASS_PATH)

from data.dataset import Dataset
import numpy as np
import pandas as pd


def read_data_file(filename: str, sep: str = ",", label: int | None = None) -> object:
    """Load data from a text file.

    Args:
        filename (str): Directory of the file.
        sep (str): Delimiters between values.
        label (int | None): Column index to be used as the dependent variable. Defaults to None.

    Returns:
        object: Dataset.
    """
    data = np.genfromtxt(filename, delimiter=sep)
    if not label:
        return Dataset(data)

    y = data[:, label]
    data = np.delete(data, label, axis=1)
    return Dataset(data, y, label)


def write_data_file(
    filename: str, dataset: object, sep: str = ",", label: bool = False
):
    """Save an array to a text file.

    Args:
        filename (str): Filename and directory desired.
        dataset (object): Dataset to convert.
        sep (str): Delimiters between values.
        label (bool): If the file has a y axis (the dependent variable).
    """
    dataset = np.append(dataset.x, dataset.y[:, None], axis=1) if not label else dataset.x
    np.savetxt(filename, X=dataset, delimiter=sep, fmt="%.3e")