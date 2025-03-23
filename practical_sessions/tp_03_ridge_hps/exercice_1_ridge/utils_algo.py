"""
Template of functions to edit to fulfill the exercises.
If you prefer, you can also start from scratch and write your own functions.
"""

import numpy as np

from constants import SIGMA


def generate_low_rank_design_matrix(n, d):
    """
    Edit this function.

    Parameters:
        n (int): number of lines of the design matrix (number of samples)
        d (int): number of columns of the design matrix (number of features)
    """
    pass


def ridge_test_error(
    n_train: int,
    d: int,
    n_repetitions: int,
    lambda_: float,
    theta_star_type: str,
    design_matrix_type: str,
) -> float:
    """
    Statistical evaluation of the excess risk of the Ridge regression
    estimator

    n_test times, do:
        - Draw output vector y, according to the linear model, fixed
        design setup.
        - compute the corresponding Ridge estimator
        - generate a test set in order to have an estimation of the excess risk of
        this estimator (generalization error)

    Parameters:
        n (int): number of samples in the dataset
        d (int): dimension of each sample (number of features)
        n_tests (int): number of simulations run, on order to
        statistically average the test errors computed and estimate
        the generalization error.

    Returns:
        risk_estimation (float): estimation of the excess risk of the OLS
        estimator in this setup.
    """
    # instantiate a PRNG
    rng = np.random.default_rng()

    return 1
