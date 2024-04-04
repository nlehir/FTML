"""
    Observe a knn adaptivity: by adaptivity, we mean that 
    the estimator automatically benefits by essence from a specific structure in the data.

    Adaptivity is a igood property of estimators, and we will
    discuss it more in the course.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from knn_1 import predict
from constants import k



def knn(n_samples: int, x_data: np.ndarray, y_data: np.ndarray, x_data_test: np.ndarray) -> float:
    """
        Compute the test error of the knn predictor
        for a given number of input samples.
    """
    return 1



def main() -> None:
    # load data
    folder = "data_knn"
    x_data = np.load(os.path.join(folder, "x_data.npy"))
    x_data_test = np.load(os.path.join(folder, "x_data_test.npy"))
    y_data = np.load(os.path.join(folder, "y_data.npy"))
    d = x_data.shape[1]
