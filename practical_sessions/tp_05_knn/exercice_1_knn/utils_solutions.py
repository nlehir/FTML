import numpy as np
from constants import k, n_test
from scipy.spatial.distance import cdist


def predict_knn(
    x_data: np.ndarray, y_data: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """
    Predict with knn estimation

    Parameters:
        x_data (float matrix): (n_samples, d) samples in input space
        y_data (float vector): (n_samples, 1) values of the target function (labels)
        (here, it is the euclidean norm, for these samples)
        x_test (float matrix): (n_test, d) data for which we
        predict a value based on the dataset.

    The samples are called "data" instead of "train" because there
    is no training involved here !

    Returns:
        y_predictions (float matrix): predictions for the data
        in x_test.
        y_predictions must be of shape (n_test,)
    """
    dist_x_data = cdist(x_test, x_data, "euclidean")
    sorted_indexes = np.argsort(dist_x_data, axis=1)
    k_neighbors = sorted_indexes[:, :k]
    y_predictions = np.mean(y_data[k_neighbors], axis=1)
    return y_predictions
