import numpy as np


def euclidean_distance(x: list, y: list) -> list:
    """Calculates the euclidean distance between the x sample and all the samples in y.

    Args:
        x (list): Unique sample.
        y (list): Array of samples.

    Returns:
        list: Euclidean distance of x sample to all the samples in y.
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
