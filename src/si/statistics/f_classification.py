import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset
from scipy import stats


def f_classification(dataset: Dataset) -> tuple:
    """Obtains the F-scores and p-values of the variables in the dataset.

    Args:
        dataset (Dataset): Dataset input.

    Returns:
        tuple: F-scores and p-values of the variables in the dataset.
    """
    classes = dataset.get_classes()
    groups = [dataset.x[dataset.y == c] for c in classes]
    f, p = stats.f_oneway(*groups)
    return f, p
