import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates the euclidean distance between the x sample and all the samples in y.

    Args:
        x (np.ndarray): Unique sample.
        y (np.ndarray): Array of samples.

    Returns:
        np.ndarray: Euclidean distance of x sample to all the samples in y.
    """
    res = []
    for sample in y:
        # Create genearator to apply in the numpy function
        generator = ((x[pos] - sample[pos]) ** 2 for pos in range(len(x)))
        # Using np.fromiter because normal np.sum is deprecated
        sum = np.sum(np.fromiter(generator, dtype=float))
        res.append(np.sqrt(sum))

    return res


if __name__ == "__main__":
    x = (1, 2, 3)
    y = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    print(euclidean_distance(x, y))
