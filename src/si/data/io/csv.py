import pandas as pd

# FIXME: What to do with features and label?
def read_csv(
    filename: str, sep: str = " ", features: bool = False, label: bool = False
) -> object:
    """Read a CSV file and transform it into a pandas dataset object.

    Args:
        filename (str): Directory of the CSV file.
        sep (str): Delimiters between values.
        features (bool): If the file has the features names.
        label (bool): If the file has a y axis.

    Returns:
        object: Dataset.
    """
    return pd.read_csv(filename, sep=sep)


# FIXME: Same as above
def write_csv(
    filename: str,
    dataset: object,
    sep: str = " ",
    features: bool = False,
    label: bool = False,
):
    """Write dataset to a CSV file specified.

    Args:
        filename (str): CSV filename and directory desired.
        dataset (object): Dataset to convert.
        sep (str): Delimiters between values.
        features (bool): If the file has the features names.
        label (bool): If the file has a y axis.
    """
    dataset.to_csv(filename, sep=sep)
