"""
    Comparison of Gradient descent (GD) and line search (LS) on a strongly convex
    loss function.
    The design matrix is randomly generated.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import OLS_estimator, gradient
from optimal_gamma import compute_gamma_star


def main() -> None:
    """
    Load the data
    """
    n = 60
    d = 40
    folder = "./data"
    X_path = os.path.join(folder, f"X_gaussian_n={n}_d={d}.npy")
    y_path = os.path.join(folder, f"y_n={n}_d={d}.npy")
    X = np.load(X_path)
    y = np.load(y_path)

    """
        Compute the OLS estimator to monitor the
        speeds of convergence
    """
    ols = OLS_estimator(X, y)

    """
        Preparation of the algorithms
    """
    theta_0 = np.zeros((d, 1))
    number_of_iterations = 5000
    GD_distances_to_opt = list()
    LS_distances_to_opt = list()

    """
    Add code here
    """


if __name__ == "__main__":
    main()
