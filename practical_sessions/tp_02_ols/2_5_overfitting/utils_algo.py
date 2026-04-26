"""
Template of function to edit to fulfill the exercise.
"""

import numpy as np
from constants import SIGMA


def ols_test_error(
    n_train: int,
    d: int,
    n_repetitions: int,
) -> float:
    """
    Statistical approximation of the generalization error (risque réel) of the OLS estimator.

    This generalization error is estimated through the computation of test
    errors, that are then averaged.

    In order to compute this test error, you will need to perform several
    steps, that include the generation of train data and the computation of the OLS
    estimator. In order to compute the OLS, you can use a manual computation using numpy,
    but you can also use scikit-learn directly if you prefer.

    Parameters:
        n (int): number of samples in the dataset
        d (int): dimension of each sample (number of features)
        n_repetitions (int): number of simulations run in order to average the
        results

    n_repetitions times, do:
        - generate a train set of length n_train, according to the linear model, fixed
        design setup.
        - compute the corresponding OLS estimator
        - generate a test set of length n_tests in order to have an estimation of the excess risk of
        this estimator (generalization error)

    Returns:
        mean_test_error (float): estimation of the generalization error of the OLS
        estimator in this statistical setting
    """
    # instantiate a PRNG
    # number of tests to estimate the excess risk
    n_tests = int(1e3)
    rng = np.random.default_rng()
    return 1
