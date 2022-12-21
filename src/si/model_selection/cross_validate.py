import sys

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

from data.dataset import Dataset
from typing import Callable


def cross_validate(
    model,
    dataset: Dataset,
    scoring: Callable = None,
    cv: int = 3,
    test_size: float = 0.2,
) -> dict:
    pass
