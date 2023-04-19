import numpy as np


def compute_gamma_star(H: np.ndarray, gradient: np.ndarray) -> float:
    """
    Exact line search gamma
    """
    square_norm = np.linalg.norm(gradient) ** 2
    d = gradient.shape[0]
    Hgrad = (H @ gradient).reshape(d)
    grad_reshape = gradient.reshape(d)
    inner_product = np.dot(Hgrad, grad_reshape)
    return square_norm / inner_product
