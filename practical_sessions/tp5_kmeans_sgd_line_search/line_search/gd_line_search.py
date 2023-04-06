"""
    Gradient descent (GD) on a strongly convex
    loss function.
    The design matrix is randomly generated.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from algorithms import OLS_estimator, gradient
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
        Compute the important quantities
    """
    # Hessian matrix
    H = 1 / n * np.matmul(np.transpose(X), X)
    # compute spectrum of H
    eigenvalues, eigenvectors = np.linalg.eig(H)
    # sort the eigenvalues
    sorted_indexes = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvectors = eigenvectors[sorted_indexes]
    # compute strong convexity and smoothness
    L = eigenvalues[0]
    mu = eigenvalues[-1]
    kappa = L / mu
    print(f"L: {L}")
    print(f"mu: {mu}")
    # OLS estimator
    eta_star = OLS_estimator(X, y)

    """
        Preparation of the algorithms
    """
    theta_0 = np.zeros((d, 1))
    number_of_iterations = 5000
    gamma_gd = 1 / L
    GD_distances_to_opt = list()
    LS_distances_to_opt = list()


if __name__ == "__main__":
    main()
