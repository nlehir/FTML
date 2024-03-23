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


def ridge_risk(n, d, lambda_, n_tests) -> float:
    """
    Statistical evaluation of the excess risk of the Ridge regression
    estimator

    In order to observe a benefit in using Ridge as compared to OLS,
    you can generate a specific design matrix X, that for instance
    has a low rank, or a d (number of features/columns) that is close
    to n (number of samples/lines).

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

    generate_low_rank_design_matrix(n=n, d=d)

    return 1
