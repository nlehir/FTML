"""
Template of functions to edit to fulfill the exercises.
If you prefer, you can also start from scratch and write your own functions.
"""

import os

import numpy as np

from constants import SIGMA


def generate_low_rank_design_matrix(n, d):
    """
    Edit this function.
    """
    pass


def ols_risk(n, d, n_tests) -> tuple[float, float]:
    """
    Statistical evaluation of the excess risk of the OLS estimator.

    n_test times, do:
        - Draw output vector y, according to the linear model, fixed
        design setup.
        - compute the corresponding OLS estimator
        - generate a test set in order to have an estimation of the excess risk of
        this estimator (generalization error)

    Parameters:
        n (int): number of samples in the dataset
        d (int): dimension of each sample (number of features)
        n_tests (int): number of simulations run

    Returns:
        risk_estimation (float): estimation of the excess risk of the OLS
        estimator in this setup.
    """
    # instantiate a PRNG
    rng = np.random.default_rng()
    return 1, 1


def ridge_risk(n, d, lambda_, n_tests) -> tuple[float, float]:
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
        n_tests (int): number of simulations run

    Returns:
        risk_estimation (float): estimation of the excess risk of the OLS
        estimator in this setup.
    """
    # instantiate a PRNG
    rng = np.random.default_rng()

    generate_low_rank_design_matrix(n=n, d=d)


    return 1
