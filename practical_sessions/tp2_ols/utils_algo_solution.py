import numpy as np

from constants import SIGMA


def generate_output_data(
    X: np.ndarray, theta_star: np.ndarray, sigma: float, rng, n_tests: int
) -> np.ndarray:
    """
    (Exact same function as in tp 2 dedicated to OLS)

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


def OLS_estimator(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute OLS estimators from the data.

    We use numpy broadcasting to accelerate computations
    and obtain several OLS estimators.

    Parameters:
        X: (n, d) matrix
        y: (n, n_tests) matrix

    We use numpy broadcasting to accelerate computations
    and actually obtain several OLS estimators (one for each column of y)

    This allows to statistically average the test errors obtained,
    ans estimator the expected value of the test error of the OLS estimator.

    Returns:
        theta_hat: (d, n_tests) matrix, one column is one OLS estimator.
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
    X = rng.uniform(low=0, high=1, size=(n, d))

    # Different design matrix
    # X = np.load("data/design_matrix.npy")
    # n, d = X.shape

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

    # compute the OLS estimator
    theta_hat = OLS_estimator(X=X, y=y_train)

    # compute the OLS regression estimators
    # there will be one different estimator per
    # output vector y
    theta_hat = OLS_estimator(
        X=X,
        y=y_train,
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

    # compute predictions of each OLS estimator
    y_pred = X @ theta_hat

    mean_test_error = np.linalg.norm(y_pred - y_test) ** 2 / (n * n_tests)

    return mean_test_error

    """
    Optional: study the variance of the relative distance
    of OLS to the bayes estimator
    """
    # distances = np.linalg.norm(theta_hat - theta_star, axis=0)
    # relative_distances = distances / np.linalg.norm(theta_star)
    # std_relative_distance = relative_distances.std()
