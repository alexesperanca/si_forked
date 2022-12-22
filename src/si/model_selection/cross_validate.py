import sys
import numpy as np

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset
from typing import Callable
from model_selection.split import train_test_split


def cross_validate(
    model: object,
    dataset: Dataset,
    scoring: Callable = None,
    cv: int = 3,
    test_size: float = 0.2,
) -> dict:
    """Perform cross validation on the given model to the inputted dataset.

    Args:
        model (object): Inputted model.
        dataset (Dataset): Dataset input.
        scoring (Callable, optional): Scoring function. Defaults to None.
        cv (int, optional): Cross validation folds. Defaults to 3.
        test_size (float, optional): Test size. Defaults to 0.2.

    Returns:
        dict: The scores of the model on the dataset
    """
    scores = {"seeds": [], "train": [], "test": []}

    for _ in range(cv):
        # Generate Random seed
        random_state = np.random.randint(0, 1000)

        # Store seed
        scores["seeds"].append(random_state)

        # Split dataset
        train, test = train_test_split(
            dataset=dataset, test_size=test_size, random_state=random_state
        )

        model.fit(train)
        if not scoring:
            scores["train"].append(model.score(train))
            scores["test"].append(model.score(test))
            continue

        # Store data
        scores["train"].append(scoring(train.y, model.predict(train)))
        scores["test"].append(scoring(test.y, model.predict(test)))
    return scores


if __name__ == "__main__":
    from io_folder.csv_file import read_csv
    from sklearn.preprocessing import StandardScaler
    from linear_model.logistic_regression import LogisticRegression

    dataset = read_csv(r"datasets\breast-bin.csv", label=-1)
    dataset.x = StandardScaler().fit_transform(dataset.x)

    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    print(cross_validate(lg, dataset, cv=5))
