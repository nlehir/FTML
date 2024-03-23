import numpy as np
from constants import SIGMA


def generate_output_data(
    X: np.ndarray, theta_star: np.ndarray, sigma: float, rng, n_tests: int
) -> np.ndarray:
    """
    generate input and output data (supervised learning)
    according to the linear model, fixed design setup
    - X is fixed
    - y is random, according to

    y = Xtheta_star + epsilon

    where epsilon is a centered gaussian noise vector with variance
    sigma*In

    We use numpy matrix manipulations in order
    to directly generate a number of output vectors.

    Parameters:
        X: (n, d) design matrix
        theta_star: (d, 1) vector (optimal parameter)
        sigma (float): variance of the noise

    Returns:
        y (float matrix): output vectors in a matrix of size (n, n_tests)
        n_tests > 1 allows to perform several tests and to statistically
        average the results.
    """
    n = X.shape[0]
    noise = rng.normal(0, sigma, size=(n, n_tests))
    y = X @ theta_star + noise
    return y


def generate_low_rank_design_matrix(n: int, d: int, rng) -> np.ndarray:
    """
    Generate a design matrix with low rank to illustrate the advantage of
    Ridge regression.
    """
    sigma_design = 1e-5
    X = rng.uniform(0, 1, size=(n, d - 1))
    X_last_column = X[:, -1].reshape(n, 1)
    noise = np.random.normal(0, sigma_design, size=(X_last_column.shape))
    X_added_column = X_last_column + noise
    X = np.hstack((X, X_added_column))
    return X


def OLS_estimator(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute OLS estimators from the data.

    We use numpy broadcasting to accelerate computations
    and obtain several OLS estimators (one for each column of y).

    Parameters:
        X: (n, d) matrix
        y: (n, n_tests) matrix

    Returns:
        theta_hat: (d, n_tests) matrix
    """
    covariance_matrix = X.T @ X
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = inverse_covariance @ (X.T @ y)
    return theta_hat


def ridge_regression_estimator(
    X: np.ndarray, y: np.ndarray, lambda_: float
) -> np.ndarray:
    """
    Compute the Ridge regression estimator

    Parameters:
        X: (n, d) matrix
        y: (n, n_tests) matrix
        lambda: regularization parameter

    We use numpy broadcasting to accelerate computations
    and obtain several Ridge estimators (one for each column of y)

    Returns:
        theta_hat: (d, n_tests) matrix
    """
    n, d = X.shape
    covariance_matrix = X.T @ X
    Sigma_matrix = covariance_matrix / n
    theta_hat = 1 / n * np.linalg.inv(Sigma_matrix + lambda_ * np.identity(d)) @ X.T @ y
    return theta_hat


def ridge_risk(n, d, lambda_, n_tests) -> float:
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

    # design matrix
    X = generate_low_rank_design_matrix(n=n, d=d, rng=rng)

    # Bayes predictor
    theta_star = rng.uniform(low=0, high=1, size=(d, 1))

    # generate train data
    # n_tests output vectors of size (n,1) are generated,
    # in order to statistically average the results
    y_train = generate_output_data(
        X=X,
        theta_star=theta_star,
        sigma=SIGMA,
        rng=rng,
        n_tests=n_tests,
    )

    # compute the Ridge regression estimators
    # there will be one different estimator per
    # output vector y
    theta_hat = ridge_regression_estimator(
        X=X,
        y=y_train,
        lambda_=lambda_,
    )

    # generate test data
    # also n_tests output vectors of size (n,1) are generated
    y_test = generate_output_data(
        X=X,
        theta_star=theta_star,
        sigma=SIGMA,
        rng=rng,
        n_tests=n_tests,
    )

    # compute predictions of each Ridge estimator
    y_pred = X @ theta_hat

    mean_test_error = np.linalg.norm(y_pred - y_test) ** 2 / (n * n_tests)

    return mean_test_error
