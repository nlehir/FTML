import numpy as np


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
    return 1/(n_samples)*(np.linalg.norm(Y-Y_predictions))**2
