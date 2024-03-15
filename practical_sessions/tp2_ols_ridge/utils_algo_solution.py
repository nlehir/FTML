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


def OLS_estimator(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute OLS estimators from the data.

    We use numpy broadcasting to accelerate computations
    and obtain several OLS estimators.

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

    # design matrix
    X = rng.uniform(0, 1, size=(n, d))

    # Different design matrix
    # X = np.load("data/design_matrix.npy")
    # n, d = X.shape

    # Bayes predictor
    theta_star = rng.uniform(0, 1, size=(d, 1))

    # run several simulations to have an estimation of the excess risk
    y = generate_output_data(X, theta_star, SIGMA, rng, n_tests)

    # compute the OLS estimator
    theta_hat = OLS_estimator(X, y)

    distances = np.linalg.norm(theta_hat - theta_star, axis=0)
    relative_distances = distances / np.linalg.norm(theta_star)
    std_relative_distance = relative_distances.std()

    # generate test data
    y_test = generate_output_data(X, theta_star, SIGMA, rng, n_tests)

    # compute predictions of each OLS estimator
    y_pred = X @ theta_hat

    mean_test_error = np.linalg.norm(y_pred - y_test) ** 2 / (n * n_tests)

    return mean_test_error, std_relative_distance
