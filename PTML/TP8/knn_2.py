import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

x_data = np.load("data_knn/x_data.npy")
x_data_test = np.load("data_knn/x_data_test.npy")
y_data = np.load("data_knn/y_data.npy")
d = x_data.shape[1]
k = 2
