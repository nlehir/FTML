"""
Generate noisy linear data with an almost colinear column in X
"""

import os

import numpy as np


def main():
    sigma = 0.01
    n = 100000
    d = 200

    rng = np.random.default_rng()

    X = rng.uniform(low=0, high=1, size=(n, d - 1))
    X_last_column = X[:, -1].reshape(n, 1)
    noise_X_last_column = rng.normal(loc=0, scale=sigma, size=(X_last_column.shape))
    X_added_column = X_last_column + noise_X_last_column
    X = np.hstack((X, X_added_column))

    theta = rng.uniform(low=0, high=1, size=(d, 1))
    noise_y = rng.normal(loc=0, scale=sigma, size=(n, 1))
    y = X @ theta + noise_y

    X_train = X[: int(n / 2)]
    X_test = X[int(n / 2) :]
    y_train = y[: int(n / 2)]
    y_test = y[int(n / 2) :]

    np.save(os.path.join("data", "X_train"), X_train)
    np.save(os.path.join("data", "X_test"), X_test)
    np.save(os.path.join("data", "y_train"), y_train)
    np.save(os.path.join("data", "y_test"), y_test)


if __name__ == "__main__":
    main()
