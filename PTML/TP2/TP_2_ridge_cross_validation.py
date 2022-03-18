import math

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt


def generate_output_data(X, theta_star, sigma):
    """
        same as previously
    """

    # output data
    n = X.shape[0]
    d = X.shape[1]
    noise = np.random.normal(0, sigma, size=(n, 1))
    Y = np.matmul(X, theta_star)+noise
    return Y


def compute_lambda_star_and_risk_star(sigma, X, theta_star, n):
    """
        same as previously
    """
    Sigma = 1/n*np.matmul(np.transpose(X), X)
    trace = np.trace(Sigma)
    llambda_star = sigma*math.sqrt(trace)/(np.linalg.norm(theta_star)*math.sqrt(n))
    excess_risk_star  = sigma**2+(sigma*math.sqrt(trace)*np.linalg.norm(theta_star))/math.sqrt(n)
    return llambda_star, excess_risk_star


def error(classifier, X, Y):
    """
        Compute the prediction error with the classifier obtained by CV,

        Parameters:
            classifier (object):
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector

        Returns:
            Mean square error
    """
    n_samples = X.shape[0]
    Y_predictions = classifier.predict(X)
    return 1/n_samples*(np.linalg.norm(Y-Y_predictions))**2


# n = 30
n = 200
# d_list = [10, 15, 20, 25, 30]
d_list = [n*10 for n in range(10, 21)]

# amount of noise (linear model, fixed design)
sigma = 0.1

llambdas_cv = list()
test_errors = list()
llambdas_star = list()
excess_risk_star = list()
