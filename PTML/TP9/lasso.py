import numpy as np

X_train = np.load("data_regression/X_train.npy")
y_train = np.load("data_regression/y_train.npy")
X_test = np.load("data_regression/X_test.npy")
y_test = np.load("data_regression/y_test.npy")
n_train, d = X_train.shape
