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

    return 1, 1
