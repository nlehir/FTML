from re import I

import numpy as np
from constants import SIGMA


def generate_output_data(
    X: np.ndarray, theta_star: np.ndarray, sigma: float, rng, n_repetitions: int
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
    noise = rng.normal(0, sigma, size=(n, n_repetitions))
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


def ridge_regression_estimator(
    X: np.ndarray, y: np.ndarray, lambda_: float
) -> np.ndarray:
    """
    Compute the Ridge regression estimators from the data.

    Parameters:
        X: (n, d) matrix
        y: (n, n_tests) matrix
        lambda: regularization parameter

    We use numpy broadcasting to accelerate computations
    and actually obtain several Ridge estimators (one for each column of y).

    This allows to statistically average the test errors obtained,
    ans estimator the expected value of the test error of the Ridge
    regression estimator.

    Returns:
        theta_hat: (d, n_tests) matrix, one column is one Ridge estimator.
    """
    n, d = X.shape
    covariance_matrix = X.T @ X
    Sigma_matrix = covariance_matrix / n
    theta_hat = 1 / n * np.linalg.inv(Sigma_matrix + lambda_ * np.identity(d)) @ X.T @ y
    return theta_hat


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
    estimator. The statistical setting is similar to the one used in OLS.
    (practical session 02)

    n_repetitions times, do:
        - Draw output vector y_train, according to the linear model, fixed
        design setup.
        - compute the corresponding Ridge estimator
        - generate a y_test in order to have an estimation of the generalization error of
        this estimator

    In the fixed design setting, which is a specific case, y_train and y_test
    contain the same number of samples, equal to n_train here.

    Parameters:
        n_train (int): number of samples in the train dataset
        d (int): dimension of each sample (number of features)
        n_repetitions (int): number of simulations run in order to average the
        results

    Returns:
        mean_test_error (float): estimation of the generalization error of the
        Ridge
        estimator in this statistical setting
    """
    # instantiate a PRNG
    rng = np.random.default_rng()

    # design matrix
    if design_matrix_type == "uniform":
        X = rng.uniform(0, 1, size=(n_train, d))
    elif design_matrix_type == "low_rank":
        X = generate_low_rank_design_matrix(n=n_train, d=d, rng=rng)
    else:
        raise ValueError("Unknown design matrix type")

    # Bayes predictor
    if theta_star_type == "random":
        theta_star = rng.uniform(low=0, high=1, size=(d, 1))
        # normalize theta_star to have a meaningful
        # comparison with the other cases: otherwise
        # the bias term direclty depends on the norm of theta_star
        # see the course for more details
        theta_star /= np.linalg.norm(theta_star)
    elif theta_star_type in ["eigenvalue_smallest", "eigenvalue_largest"]:
        Sigma = X.T @ X
        # all the eigenvalues are >= 0 (to prove in exercise)
        # the eigenvector are sorted in decreasing eigenvalue
        _, eigenvectors = np.linalg.eig(Sigma)
        if theta_star_type == "eigenvalue_smallest":
            theta_star = eigenvectors[:, -1].reshape(d, 1)
        else:
            theta_star = eigenvectors[:, 0].reshape(d, 1)
    else:
        raise ValueError("Unknown eigenvector type")

    # generate train data
    # n_tests output vectors of size (n,1) are generated,
    # in order to statistically average the results
    y_train = generate_output_data(
        X=X,
        theta_star=theta_star,
        sigma=SIGMA,
        rng=rng,
        n_repetitions=n_repetitions,
    )

    # compute the Ridge regression estimators
    # there will be one different estimator per
    # output vector y
    theta_hat = ridge_regression_estimator(
        X=X,
        y=y_train,
        lambda_=lambda_,
    )
    # if n_repetitions > 10:
    # __import__('ipdb').set_trace()

    # generate test data
    # also n_tests output vectors of size (n,1) are generated
    y_test = generate_output_data(
        X=X,
        theta_star=theta_star,
        sigma=SIGMA,
        rng=rng,
        n_repetitions=n_repetitions,
    )

    # compute predictions of each Ridge estimator
    y_pred = X @ theta_hat

    n_test = n_train
    mean_test_error = (np.linalg.norm(y_pred - y_test) ** 2 / n_test) / n_repetitions

    return mean_test_error
