"""
    observe the adaptivity to a low dimensional support.
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
        Run knn with a n_samples
    """
    return 1



def main() -> None:
    # load data
    folder = "data_knn"
    x_data = np.load(os.path.join(folder, "x_data.npy"))
    x_data_test = np.load(os.path.join(folder, "x_data_test.npy"))
    y_data = np.load(os.path.join(folder, "y_data.npy"))
    d = x_data.shape[1]
