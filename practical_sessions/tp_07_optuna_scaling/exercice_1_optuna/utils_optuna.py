import numpy as np


def prediction_squared_error(estimator, X, y) -> float:
    predictions = estimator.predict(X)
    n_samples = X.shape[0]
    error = predictions - y
    return np.linalg.norm(error) ** 2 / n_samples
