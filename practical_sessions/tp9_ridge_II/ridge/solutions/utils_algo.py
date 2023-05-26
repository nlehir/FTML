import numpy as np
import math
from constants import rng, SIGMA, N_TESTS


def generate_output_data(
    X: np.ndarray, theta_star: np.ndarray, sigma: float, n_tests: int
) -> np.ndarray:
    """
    generate input and output data (supervised learning)
    according to the linear model, fixed design setup
    - X is fixed
    - y is random, according to

    Y = Xtheta_star + epsilon

    We use numpy matrix manipulations in order
    to directly generate a number of output vectors.

    where epsilon is a centered gaussian noise vector with variance
    sigma*In

    Parameters:
        X: (n, d) design matrix
        theta_star: (d, 1) vector (optimal parameter)
        sigma (float): variance of the noise

    Returns:
        Y (float matrix): output vector (n, 1)
    """
    n = X.shape[0]
    noise = rng.normal(0, sigma, size=(n, n_tests))
    y = X @ theta_star + noise
    return y


def ridge_regression_estimator(
    X: np.ndarray, y: np.ndarray, lambda_: float
) -> np.ndarray:
    """
    Compute the Ridge regression estimator

    We use numpy broadcasting to accelerate computations
    and obtain several OLS estimators.

    Parameters:
        X: (n, d) matrix
        y: (n, n_tests) matrix
        lambda: regularization parameter

    Returns:
        theta_hat: (d, n_tests) matrix
    """
    n, d = X.shape
    covariance_matrix = X.T @ X
    Sigma_matrix = covariance_matrix / n
    theta_hat = 1 / n * np.linalg.inv(Sigma_matrix + lambda_ * np.identity(d)) @ X.T @ y
    return theta_hat


def error(theta: np.ndarray, X: np.ndarray, targets: np.ndarray) -> float:
    """
        Compute the prediction error with parameter theta,
        between Xtheta and the labels Y

        Parameters:
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector
            theta (float vector): (d, 1) vector
        Returns:
            Mean square error
    """
    n_samples = X.shape[0]
    preds = X @ theta
    return 1/n_samples*(np.linalg.norm(preds-targets))**2


def compute_lambda_star_and_risk_star(sigma: float, X: np.ndarray, theta_star: np.ndarray) -> tuple[float, float]:
    """
        Compute lambda_star for which we have theoretical
        garantees on the value of the excess risk.

        Parameters:
            sigma (float): variance of the linear model, fixed design
            X (float matrix): (n, d) matrix
            theta_star (float vector): (d, 1) optimal parameter (Bayes
            predictor)
            n (int): number of samples

        Returns:
            llambda_star (float)
            risk_star (float)

    """
    # print(f"n={n}, d={d}, trace={trace}")
    n = X.shape[0]
    Sigma = 1/n*X.T @ X
    trace = np.trace(Sigma)
    llambda_star = sigma*math.sqrt(trace)/(np.linalg.norm(theta_star)*math.sqrt(n))
    risk_star  = sigma**2+(sigma*math.sqrt(trace)*np.linalg.norm(theta_star))/math.sqrt(n)
    return llambda_star, risk_star


def ridge_risk(
        lambda_: float,
        n_tests: int,
        theta_star: np.ndarray,
        X: np.ndarray,
        ) -> float:
    """
    Statistical evaluation of the excess risk of the Ridge regression
    estimator

    n_test times, do:
        - Draw an output vector y, according to the linear model, fixed
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
    n = X.shape[0]

    # run several simulations to have an estimation of the excess risk
    y_train = generate_output_data(X, theta_star, SIGMA, N_TESTS)

    # compute the Ridge regression estimator
    theta_hat = ridge_regression_estimator(X, y_train, lambda_)

    # generate test data
    y_test = generate_output_data(X, theta_star, SIGMA, N_TESTS)

    # compute predictions of each OLS estimator
    y_pred = X @ theta_hat

    # pred_size = len(y_pred) 
    mean_test_error = np.linalg.norm(y_pred - y_test) ** 2 / (n * n_tests)

    return mean_test_error
