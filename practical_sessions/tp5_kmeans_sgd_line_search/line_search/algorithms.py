import math

import numpy as np


def OLS_estimator(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute OLS estimators from the data.

    We use numpy broadcasting to accelerate computations
    and obtain several OLS estimators.

    Parameters:
        X: (n, d) matrix
        y: (n, n_tests) matrix

    Returns:
        theta_hat: (d, n_tests) matrix
    """
    covariance_matrix = X.T @ X
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = inverse_covariance @ (X.T @ y)
    return theta_hat


def error(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the prediction error with parameter theta,
    between Xtheta and the labels y

    Parameters:
        X: (n, d) matrix
        y: (n, 1) vector
        theta: (d, 1) vector

    Returns:
        Mean square error
    """
    n_samples = X.shape[0]
    y_predictions = X @ theta
    return 1 / (2 * n_samples) * (np.linalg.norm(y - y_predictions)) ** 2


def gradient(theta, H, X, y):
    """
    Compute the gradient of the empirical risk
    as a function of theta, X, y
    for a least squares problem.

    Parameters:
        X (float matrix): (n, d) matrix
        y (float vector): (n, 1) vector
        theta (float vector): (d, 1) vector

    Returns:
        gradient of the objective function
    """
    n = y.shape[0]
    return H @ theta - 1 / n * X.T @ y
