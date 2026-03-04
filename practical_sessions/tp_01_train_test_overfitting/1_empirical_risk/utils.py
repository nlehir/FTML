"""
Utility functions for 1d linear regression
"""

import numpy as np


def empirical_risk(
    theta: float,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Compute the empirical risk of a set of parameters
    for a linear prediction, in 1 dimension.

    fix this function
    """
    return 1


def compute_optimal_params(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Compute the optimal theta and b, obtained by
    gradient cancellation
    """
    n_samples = X.shape[0]

    # intermediate quantities
    sum_xy = np.sum(X * y)
    sum_xx = np.sum(X * X)
    sum_x_2 = np.sum(X) ** 2
    sum_y = np.sum(y)
    sum_x = np.sum(X)

    # apply formulas for gradient cancellation
    theta_star = (sum_xy - sum_x * sum_y / n_samples) / (sum_xx - sum_x_2 / n_samples)
    b_star = (sum_y - theta_star * sum_x) / n_samples

    return theta_star, b_star
