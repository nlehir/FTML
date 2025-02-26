"""
Study overfitting and variance of the test error estimation
by monitoring the R2 train and test scores after subsampling the datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression


def main():
    X_train = np.load(os.path.join("data", "X_train.npy"))
    X_test = np.load(os.path.join("data", "X_test.npy"))
    y_train = np.load(os.path.join("data", "y_train.npy"))
    y_test = np.load(os.path.join("data", "y_test.npy"))

    rng = np.random.default_rng()

    """
    Study the variance of the test error estimation
    by subsampling the test set
    n_train will also have an influence on the result
    """

    """
    Study overfitting
    by subsampling the train set
    """


if __name__ == "__main__":
    main()
