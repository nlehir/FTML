import numpy as np


def add_intercept(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    return np.hstack((data, np.ones((n, 1))))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid function elementwise to the array
    """
    return 1 / (1 + np.exp(-z))


def sign(x) -> float:
    if x > 0:
        return 1
    else:
        return 0


def empirical_risk(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    n_samples = X.shape[0]
    estimations = (X @ theta).reshape(n_samples, 1)
    losses = np.log(1 + np.exp(-estimations * y))
    emp_risk = losses.mean()
    return emp_risk


def compute_accuracy(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    estimates = X @ theta
    products = estimates * y
    signs = np.sign(products)
    losses = (signs + 1) / 2
    accuracy = losses.mean()
    return accuracy
