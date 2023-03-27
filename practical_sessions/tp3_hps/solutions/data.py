"""
Preprocess data before learning

Used by train_validation_test.py and cross_validation.py
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.utils import shuffle

# digits
# 250 points
# wine
# nb_points = 60

dataset = "wine"
nb_points = 100
print(f"\ndataset: {dataset}")


def load_data(dataset) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "digits":
        data = load_digits()
    elif dataset == "iris":
        data = load_iris()
    elif dataset == "wine":
        data = load_wine()
    elif dataset == "breast_cancer":
        data = load_breast_cancer()
    X, y = data.data, data.target
    return X, y


def process_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle and downsample the data
    in order to observe the statistical behavior of the HP validation
    procedures investigated.
    """

    X, y = shuffle(X, y)
    # downsample data
    n_samples = len(X)
    # print(f"{n_samples} samples")
    if nb_points < n_samples:
        X = X[:nb_points]
        y = y[:nb_points]
    return X, y
