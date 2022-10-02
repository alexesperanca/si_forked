import numpy as np

# FIXME: Label?
def read_data_file(filename: str, sep: str = " ", label: bool = False) -> object:
    """Load data from a text file.

    Args:
        filename (str): Directory of the file.
        sep (str): Delimiters between values.
        label (bool): If the file has a y axis.

    Returns:
        object: Dataset.
    """
    return np.genfromtxt(filename, delimiter=sep)


def write_data_file(
    filename: str, dataset: object, sep: str = " ", label: bool = False
):
    """Save an array to a text file.

    Args:
        filename (str): Filename and directory desired.
        dataset (object): Dataset to convert.
        sep (str): Delimiters between values.
        label (bool): If the file has a y axis.
    """
    np.savetxt(filename, dataset, delimiter=sep)
