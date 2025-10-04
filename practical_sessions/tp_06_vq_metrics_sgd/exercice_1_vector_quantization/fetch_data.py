"""
Fetch MNIST dataset
Save a subset of the dataset to a local file.
"""

import os

import numpy as np
import sklearn.datasets


def main():
    X, y = sklearn.datasets.fetch_openml("mnist_784", version=1, return_X_y=True)

    X_train = X[:10000]
    y_train = y[:10000].astype(int)
    X_test = X[10000:20000]
    y_test = y[10000:20000].astype(int)
    np.save(os.path.join("data", "X_train"), X_train)
    np.save(os.path.join("data", "y_train"), y_train)
    np.save(os.path.join("data", "X_test"), X_test)
    np.save(os.path.join("data", "y_test"), y_test)


if __name__ == "__main__":
    main()
