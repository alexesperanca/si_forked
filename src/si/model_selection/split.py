import sys
import numpy as np

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset


def train_test_split(dataset: object, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Division of the given data into a section to train and other to test.

    Args:
        dataset (object): Dataset class instance.
        test_size (float): Size of the division to test the data.
        random_state (int): Seed value to reproduce a result. Change this value if the sample randomize should be renewed.

    Returns:
        tuple: Dataset class instances, one being the train data and the other the test data.
    """
    # If the user does not pass a value between 0 and 1
    if test_size > 1:
        test_size = test_size / 100

    # Makes the seed choice aleatory
    np.random.seed(random_state)

    dataset_size = dataset.shape()[0][0]
    index_division = round(dataset_size * test_size)

    # Generate the permutations
    permutations = np.random.permutation(dataset_size)

    # Get the test and train samples
    test_index, train_index = (
        permutations[:index_division],
        permutations[index_division:],
    )

    train_x, test_x, train_y, test_y = (
        dataset.x[train_index],
        dataset.x[test_index],
        dataset.y[train_index] if dataset.y else None,
        dataset.y[test_index] if dataset.y else None,
    )

    train_data = Dataset(train_x, train_y, dataset.features, dataset.label)
    test_data = Dataset(test_x, test_y, dataset.features, dataset.label)
    return train_data, test_data
