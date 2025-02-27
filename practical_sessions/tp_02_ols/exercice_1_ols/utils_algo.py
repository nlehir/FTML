"""
Template of functions to edit to fulfill the exercises.
If you prefer, you can also start from scratch and write your own functions.
"""

import os

import numpy as np


def ols_test_error(n, d) -> float:
    """
    Statistical approximation of the generalization error (risque réel) of the OLS estimator.

    This generalization error is estimated through the computation of a test
    error.

    In order to compute this test error, you will need to perform several
    steps, that include the generation of data and the computation of the OLS
    estimator. In order to compute the correct quantity, you will need to be
    careful about the usage of these data. In order to compute the OLS, you can
    use a manual computation using numpy, but you can also use scikit-learn
    directly if you prefer.

    Parameters:
        n (int): number of samples in the dataset
        d (int): dimension of each sample (number of features)

    Returns:
        risk_estimation (float): estimation of the generalization error of the OLS
        estimator in this statistical setup.

    """
    # instantiate a PRNG
    # number of tests to estimate the excess risk
    n_tests = int(1e3)
    rng = np.random.default_rng()
    return 1
