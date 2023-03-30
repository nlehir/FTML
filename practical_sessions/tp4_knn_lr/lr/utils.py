import numpy as np


def compute_gamma_star(H, gradient) -> float:
    """
    Line search gamma
    """
    square_norm = np.linalg.norm(gradient) ** 2
    d = gradient.shape[0]
    Hgrad = np.matmul(H, gradient).reshape(d)
    grad_reshape = gradient.reshape(d)
    inner_product = np.dot(Hgrad, grad_reshape)
    return square_norm / inner_product


def add_intercept(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    return np.hstack((data, np.ones((n, 1))))


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sign(x) -> float:
    if x > 0:
        return 1
    else:
        return 0


def empirical_risk(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    n_samples = X.shape[0]
    estimations = (X @ theta).reshape(n_samples, 1)
    losses = np.log(1 + np.exp(-estimations * y))
    emp_risk = 1 / n_samples * losses.sum()
    return emp_risk


def compute_accuracy(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    estimates = X @ theta
    products = estimates * y
    signs = np.sign(products)
    losses = (signs + 1) / 2
    accuracy = losses.mean()
    return accuracy
