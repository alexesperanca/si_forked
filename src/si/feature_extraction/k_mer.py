import sys
import itertools

CLASSES_PATH = "src/si"
sys.path.insert(0, CLASSES_PATH)

import numpy as np
from data.dataset import Dataset


class KMer:
    def __init__(self, k: int = 2, alphabet: str = "ACTG"):
        self.k = k
        self.k_mers = None
        self.alphabet = alphabet

    def fit(self) -> "KMer":
        # Generate the KMers -> All possible combinations of the input sequence of length self.k
        self.k_mers = [
            "".join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)
        ]
        return self

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i : i + self.k]
            counts[k_mer] += 1

        # Normalize the counts
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        # Calculate the k-mer composition
        sequences_k_mer_composition = [
            self._get_sequence_k_mer_composition(sequence)
            for sequence in dataset.x[:, 0]
        ]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        return Dataset(
            sequences_k_mer_composition,
            dataset.y,
            self.k_mers,
            dataset.label,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        return self.fit().transform(dataset)


if __name__ == "__main__":
    from data.dataset import Dataset
    from model_selection.split import train_test_split
    from sklearn.preprocessing import StandardScaler
    from io_folder.csv_file import read_csv
    from linear_model.logistic_regression import LogisticRegression

    dataset = read_csv(r"datasets/tfbs.csv", label=-1)

    k_mer_ = KMer(k=3)
    dataset = k_mer_.fit_transform(dataset)
    print(dataset.x)
    print(dataset.features)
    
    dataset.x = StandardScaler().fit_transform(dataset.x)

    # Split the dataset
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)
    lg = LogisticRegression()
    lg.fit(dataset, use_adaptive_alpha=True)
    print("Score:", lg.score(dataset))
    print("Cost:", lg.cost(dataset))
    lg.cost_plot()
    
    print("\nNew Test")
    dataset_trans = read_csv(r"datasets/transporters.csv", label=-1)

    k_mer_transp = KMer(k=2, alphabet="ACDEFGHIKLMNPQRSTVWY")
    dataset_trans = k_mer_transp.fit_transform(dataset_trans)
    print(dataset_trans.x)
    print(dataset_trans.features)
    
    # Split the dataset
    dataset_transp_train, dataset_transp_test = train_test_split(dataset_trans, test_size=0.2)
    lg = LogisticRegression()
    lg.fit(dataset_trans)
    print("Score:", lg.score(dataset_trans))
    print("Cost:", lg.cost(dataset_trans))
    lg.cost_plot()
    
