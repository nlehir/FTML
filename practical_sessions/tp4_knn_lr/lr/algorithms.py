"""
Functions used by first-order methods
These functions need to be fixed.
"""


import numpy as np

from utils import empirical_risk, sigmoid

conv_tolerance = 1.05


def gradient_one_sample(theta, x, y, mu) -> np.ndarray:
    return theta


def batch_gradient(theta, X, Y, mu) -> np.ndarray:
    return theta


def GD(
    gamma,
    X_train,
    y_train,
    n_iterations,
    scikit_empirical_risk,
    mu,
) -> np.ndarray:
    print("\nGD")
    d = X_train.shape[1]
    theta_gd = np.zeros((d, 1))
    return theta_gd


def SGD(
    gamma_0,
    X_train,
    y_train,
    n_iterations,
    scikit_empirical_risk,
    mu,
    schedule,
) -> np.ndarray:
    print("\nSGD")
    d = X_train.shape[1]
    theta_sgd = np.zeros((d, 1))
    return theta_sgd
