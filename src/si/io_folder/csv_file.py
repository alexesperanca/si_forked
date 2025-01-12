import sys

DATASET_CLASS_PATH = "src/si"
sys.path.insert(0, DATASET_CLASS_PATH)

import pandas as pd
import numpy as np
from data.dataset import Dataset


def read_csv(
    filename: str, sep: str = ",", features: bool = False, label: int = None
) -> Dataset:
    """Read a CSV file and transform it into a pandas dataset object.

    Args:
        filename (str): Directory of the CSV file.
        sep (str): Delimiters between values. Defaults to ",".
        features (bool): If the file has the features names. Defaults to False.
        label (int | None): Column index to be used as the dependent variable. Defaults to None.

    Returns:
        Dataset: Final Dataset read from CSV.
    """
    data = pd.read_csv(filename, sep=sep)
    headers = list(data.columns)
    header_label = headers[label] if label else None
    new_features = []

    if label and features:
        new_features = [col for idx, col in enumerate(data.columns) if idx != label]
    elif features:
        new_features = list(data.columns)

    data = data.to_numpy()
    if not label:
        return Dataset(data, None, new_features, header_label)

    # Define the column defined as the Y data
    y = data[:, label]
    data = np.delete(data, label, axis=1)
    return Dataset(data, y, new_features, header_label)


def write_csv(
    filename: str,
    dataset: object,
    sep: str = ",",
    features: bool = True,
    label: bool = False,
):
    """Write dataset to a CSV file specified.

    Args:
        filename (str): CSV filename and directory desired.
        dataset (object): Dataset to convert.
        sep (str): Delimiters between values.
        features (bool): If the file has the features names. Defaults to True.
        label (bool): If the file has a y axis. Defaults to False.
    """
    data = pd.DataFrame(dataset.x)
    if features:
        data.columns = dataset.features

    if label:
        data.insert(loc=0, column=dataset.label, value=dataset.y)

    data.to_csv(filename, sep=sep, index=False)
