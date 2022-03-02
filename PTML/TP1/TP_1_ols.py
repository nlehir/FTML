import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# if you want to clean the images
# image_directory = "./images_ols"
# for image in os.listdir(image_directory):
#     os.remove(os.path.join(image_directory, image))

"""
    This scripts studies the OLS estimator:
        - risk
        - dependence of the risk on the dimensions n and d
        - stability of the OLS estimator
"""


def generate_output_data(X, theta_star, sigma):
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

        FIX THIS FUNCTION
    """

    # output data
    n = X.shape[0]
    Y = np.zeros((n, 1))
    return Y


def OLS_estimator(X, Y):
    """
        Compute the OLS estimator from the data.

        Parameters:
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector

        Returns:
            OLS (float vector): (d, 1) vector

        FIX THIS FUNCTION
    """
    return np.ones((n, 1))


def distance_to_theta_star(theta_hat, theta_star):
    """
        Relative distance between theta_hat and theta_star
    """
    return np.linalg.norm(theta_hat - theta_star)/np.linalg.norm(theta_star)


def error(theta_hat, X, Y):
    """
        Compute the prediction error with parameter theta_hat,
        between Xtheta_hat and the labels Y

        Parameters:
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector
            theta_hat (float vector): (d, 1) vector

        Returns:
            Mean square error

        FIX THIS FUNCTION
    """
    return 1


def ols_risk(n, d, n_tests):
    """
        Statistical evaluation of the excess risk of the OLS estimator.
        n_test times, do:
            - Draw output vector Y, according to the linear model, fixed
            design setup.
            - split the dataset into a train set and a test set
            - use the train set compute the OLS estimator
            - use the test test in order to have an estimation of the risk of
            this estimator (generalization error)

        In order to test the estimator and estimate the risk,
        we need to use a different output vector Y_test,
        that has not been used for training.

        Parameters:
            n (int): number of samples in the dataset
            d (int): dimension of each sample (number of features)
            n_tests (int): number of simulations run

        Returns:
            risk_estimation (float): estimation of the excess risk of the OLS
            estimator in this setup.

    """
    # design matrix
    X = r.rand(n, d)
    # Different design matrix
    # X = np.load("data/design_matrix.npy")
    # n, d = X.shape

    # Bayes predictor
    theta_star = r.rand(d).reshape(d, 1)

    Y = generate_output_data(X, theta_star, sigma)
    # compute the OLS estimator
    theta_hat = OLS_estimator(X, Y)
    # compute and store empirical risk (train error)
    empirical_risks = error(theta_hat, X, Y)
    # generate test data
    nb_test_samples = int(n/2)
    X_test = X[:nb_test_samples, :]
    Y_test = generate_output_data(X_test, theta_star, sigma)
    # compute and store test error
    test_error = error(theta_hat, X_test, Y_test)

    return test_error


# dimensions of the problem
n = 30
d = 10

# amount of noise (linear model, fixed design)
sigma = 0.2
bayes_risk = sigma**2

# use a seed to have consistent resutls
r = np.random.RandomState(4)

# number of tests to estimate the excess risk
n_tests = 200

test_error = ols_risk(n, d, n_tests)
print(f"n={n}, d={d}")
print(f"test error: {test_error}")
