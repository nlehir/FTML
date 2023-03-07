import numpy as np
import os
from constants import SIGMA


def ols_risk(n, d, n_tests) -> tuple[float, float]:
    """
        Statistical evaluation of the excess risk of the OLS estimator.

        n_test times, do:
            - Draw output vector Y, according to the linear model, fixed
            design setup.
            - compute the corresponding Ridge estimator
            - generate a test test in order to have an estimation of the excess risk of
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

        n_test times, do:
            - Draw output vector Y, according to the linear model, fixed
            design setup.
            - compute the corresponding Ridge estimator
            - generate a test test in order to have an estimation of the excess risk of
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

    # use a specific design matrix
    # data_path = os.path.join("data", f"design_matrix_n={n}_d={d}.npy")
    # if not os.path.exists(data_path):
    #     print("generate matrix")
    #     X = generate_low_rank_design_matrix(n, d, rng)
    # else:
    #     X = np.load(data_path)

    return 1
