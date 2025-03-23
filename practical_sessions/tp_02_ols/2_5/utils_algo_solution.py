import numpy as np

from constants import SIGMA


def generate_output_data(
    X: np.ndarray, theta_star: np.ndarray, sigma: float, rng, n_repetitions: int
) -> np.ndarray:
    """
    Generate output data according to the linear model, fixed design setup
    - X is fixed
    - y is random, according to

    y = Xtheta_star + epsilon

    where epsilon is a centered gaussian noise vector with variance
    sigma*In

    Parameters:
        X: (n, d) design matrix
        theta_star: (d, 1) vector (optimal parameter)
        sigma (float): variance of the noise

    Returns:
        y (float matrix): output vectors in a matrix of size (n, n_repetitions)
        n_repetitions > 1 allows to perform several tests and to statistically
        average the results.
    """
    n = X.shape[0]
    noise = rng.normal(0, sigma, size=(n, n_repetitions))
    y = X @ theta_star + noise
    return y


def OLS_estimator(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute OLS estimators from the data.

    We obtain as many OLS estimators as there are columns in y.
    This is useful in this exercice in order to average the results
    of the test errors afterwards.

    Parameters:
        X: (n, d) matrix
        y: (n, n_repetitions) matrix

    Returns:
        theta_hat: (d, n_repetitions) matrix, one column is one OLS estimator.
    """
    covariance_matrix = X.T @ X
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = inverse_covariance @ (X.T @ y)
    return theta_hat


def ols_test_error(
    n_train: int,
    d: int,
    n_repetitions: int,
) -> float:
    """
    Statistical evaluation of the generalization error of the OLS estimator.
    The generalization error is an expected value. We approximate this
    expected value by sampling some data, computing test errors, and repeting the simulation a
    number of times that is approximately sufficient to observe convergence.

    n_repetitions times, do:
        - Draw output vector y_train, according to the linear model, fixed
        design setup.
        - compute the corresponding OLS estimators. Each OLS estimator will be
          different.
        - generate a y_test in order to have an estimation of the generalization error of
        this estimator

    Importantly, our expected value is taken over the whole data generation
    process. As the OLS estimator depends on the y_train, several OLS
    estimators will be computed.

    In the fixed design setting, which is a specific case, y_train and y_test
    contain the same number of samples, equal to n_train here.

    Parameters:
        n_train (int): number of samples in the train dataset
        d (int): dimension of each sample (number of features)
        n_repetitions (int): number of simulations run in order to average the
        results

    Returns:
        mean_test_error (float): estimation of the generalization error of the OLS
        estimator in this statistical setting
    """
    # instantiate a PRNG
    rng = np.random.default_rng()

    # design matrix
    X = rng.uniform(low=0, high=1, size=(n_train, d))

    # Bayes predictor
    theta_star = rng.uniform(low=0, high=1, size=(d, 1))

    # generate train data
    # n_repetitions output vectors of size (n,1) are generated,
    # in order to statistically average the results
    y_train = generate_output_data(
        X=X,
        theta_star=theta_star,
        sigma=SIGMA,
        rng=rng,
        n_repetitions=n_repetitions,
    )

    # compute the OLS regression estimators
    # there will be one different estimator per
    # output vector y
    theta_hat = OLS_estimator(
        X=X,
        y=y_train,
    )

    # generate test data
    # also n_repetitions output vectors of size (n,1) are generated
    y_test = generate_output_data(
        X=X,
        theta_star=theta_star,
        sigma=SIGMA,
        rng=rng,
        n_repetitions=n_repetitions,
    )

    # compute predictions of each OLS estimator
    y_pred = X @ theta_hat

    """
    Compute the test erros on each y_test, and
    average the results in order to have an approximation
    of the expected value.
    """
    n_test = n_train
    mean_test_error = (np.linalg.norm(y_pred - y_test) ** 2 / n_test) / n_repetitions

    return mean_test_error

    """
    Optional: study the variance of the relative distance
    of OLS to the bayes estimator
    """
    # distances = np.linalg.norm(theta_hat - theta_star, axis=0)
    # relative_distances = distances / np.linalg.norm(theta_star)
    # std_relative_distance = relative_distances.std()
