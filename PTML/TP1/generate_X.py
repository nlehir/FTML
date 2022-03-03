import numpy as np

sigma = 0.005
n = 30
d = 10

X = np.random.rand(n, d-1)
X_last_column = X[:, -1].reshape(n, 1)
noise = np.random.normal(0, sigma, size=(X_last_column.shape))
X_added_column = X_last_column + noise
X = np.hstack((X, X_added_column))
np.save("data/design_matrix", X)
