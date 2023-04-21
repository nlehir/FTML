import numpy as np
import math


def error(theta, X, Y):
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
    y_predictions = X @ theta
    return 1 / (2 * n_samples) * (np.linalg.norm(y - y_predictions)) ** 2


def gradient(theta, H, X, y):
    """
    Compute the gradient of the empirical risk
    as a function of theta, X, y

    Parameters:
        X (float matrix): (n, d) matrix
        Y (float vector): (n, 1) vector
        theta (float vector): (d, 1) vector

    Returns:
        gradient of the objective function
    """
    n = y.shape[0]
    return H @ theta - 1 / n * X.T @ y


def upper_bound_strongly_convex(t, kappa, theta_0, theta_star):
    initial_square_norm = np.linalg.norm(theta_0 - theta_star) ** 2
    rate = math.exp(-2 * t / kappa)
    return initial_square_norm * rate


def generate_output_data(X, theta_star, sigma, r):
    """
    generate input and output data according to
    the linear model, fixed design setup
    - X is fixed
    - y is random, according to

    y = Xtheta_star + epsilon

    where epsilon is a centered gaussian noise vector with variance
    sigma*In

    Parameters:
        X (float matrix): (n, d) design matrix
        theta_star (float vector): (d, 1) vector (optimal parameter)
        sigma (float): variance each epsilon

    Returns:
        y (float matrix): output vector (n, 1)
    """

    # output data
    n = X.shape[0]
    noise = r.normal(0, sigma, size=(n, 1))
    y = X @ theta_star + noise
    return y
