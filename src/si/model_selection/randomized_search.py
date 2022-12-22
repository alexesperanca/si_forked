import sys
import numpy as np

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset
from typing import Callable
from model_selection.cross_validate import cross_validate


def randomized_search_cv(
    model: object,
    dataset: Dataset,
    parameter_distribution: dict,
    scoring: Callable = None,
    cv: int = 5,
    n_iter: int = 10,
    test_size: float = 0.2,
) -> dict:
    """Cross validation of the model with a random parameter value chosen.
    Process runs according to the number of iterations inputted.

    Args:
        model (object): Inputted model.
        dataset (Dataset): Dataset input.
        parameter_distribution (dict): The parameter distribution to use.
        scoring (Callable, optional): Scoring function. Defaults to None.
        cv (int, optional): Cross validation folds. Defaults to 5.
        n_iter (int, optional): Number of iterations. Defaults to 10.
        test_size (float, optional): Test size. Defaults to 0.2.

    Returns:
        dict: The scores of the model on the dataset.
    """
    # Validate the parameter grid
    for parameter in parameter_distribution:
        # Verifies if the model has the attribute parameter
        assert hasattr(
            model, parameter
        ), f"Model {model} does not have parameter {parameter}."

    scores = []
    for _ in range(n_iter):
        params = {}
        for param, values in parameter_distribution.items():
            random_value = np.random.choice(values)
            params[param] = random_value
            setattr(model, parameter, random_value)

        # Cross validate the model
        score = cross_validate(model, dataset, scoring, cv, test_size)

        # Add the parameter configuration
        score["parameters"] = params
        scores.append(score)
    return scores


if __name__ == "__main__":
    from io_folder.csv_file import read_csv
    from sklearn.preprocessing import StandardScaler
    from linear_model.logistic_regression import LogisticRegression

    dataset = read_csv(r"datasets\breast-bin.csv", label=-1)
    dataset.x = StandardScaler().fit_transform(dataset.x)

    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    grid = {"l2_penalty": (1, 10), "alpha": (0.001, 0.0001), "max_iter": (1000, 2000)}
    print(randomized_search_cv(lg, dataset, grid))
