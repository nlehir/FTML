import numpy as np
import math


def OLS_estimator(X, y):
    """
        Compute the OLS estimator from the data.

        Parameters:
            X (float matrix): (n, d) matrix
            y (float vector): (n, 1) vector

        Returns:
            OLS (float vector): (d, 1) vector
    """
    covariance_matrix = np.matmul(np.transpose(X), X)
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = np.matmul(inverse_covariance, np.matmul(np.transpose(X), y))
    return theta_hat


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
    Y_predictions = np.matmul(X, theta)
    return 1/(2*n_samples)*(np.linalg.norm(Y-Y_predictions))**2


def loss_ridge(theta, X, Y, nu):
    """
        Compute the ridge regression loss with parameter theta,
        between Xtheta and the labels Y

        Parameters:
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector
            theta (float vector): (d, 1) vector

        Returns:
            Ridge risk
    """
    n_samples = X.shape[0]
    Y_predictions = np.matmul(X, theta)
    return 1/(2*n_samples)*(np.linalg.norm(Y-Y_predictions))**2+(nu/2)*np.linalg.norm(theta)


def gradient(theta, H, X, y):
    """
        EDIT THIS FUNCTION

        Compute the gradient of the empirical risk
        as a function of theta, X, y

        Parameters:
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector
            theta (float vector): (d, 1) vector

        Returns:
            gradient of the objective function
    """
    return theta


def upper_bound_strongly_convex(t, kappa, theta_0, eta_star):
    """
        EDIT THIS FUNCTION

        Compute the upper bound of the convergence rate
        for a strongly convex function
    """
    return 1


def compute_gamma_star(H, gradient):
    """
        EDIT THIS FUNCTION

        Line search gamma
    """
    return 1


def upper_bound_convex(t, kappa, theta_0, eta_star, gamma):
    """
        EDIT THIS FUNCTION

        Compute the upper bound of the convergence rate
        for a strongly convex function
    """
    return 1


def square_distance_to_optimal_parameter(theta, eta_star):
    """
        EDIT THIS FUNCTION
    """
    return 1


def generate_output_data(X, theta_star, sigma, r):
    """
        generate input and output data (supervised learning)
        according to the linear model, fixed design setup
        - X is fixed
        - Y is random, according to

        Y = Xtheta_star + epsilon

        where epsilon is a centered gaussian noise vector with variance
        sigma*In

        Parameters:
            X (float matrix): (n, d) design matrix
            theta_star (float vector): (d, 1) vector (optimal parameter)
            sigma (float): variance each epsilon

        Returns:
            Y (float matrix): output vector (n, 1)
    """

    # output data
    n = X.shape[0]
    d = X.shape[1]
    noise = r.normal(0, sigma, size=(n, 1))
    Y = np.matmul(X, theta_star)+noise
    return Y
