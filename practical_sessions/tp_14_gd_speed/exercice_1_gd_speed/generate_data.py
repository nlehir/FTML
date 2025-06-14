import math

import numpy as np

from params import N, D

n_list = [N]
d_list = [D]


for n in n_list:
    for d in d_list:
        # random invertible
        X = np.random.rand(n, d)
        np.save(f"data/X_gaussian_n={n}_d={d}", X)

        # diagonal
        X = np.zeros((n, d))
        for j in range(d):
            X[j, j] = 1/(j+1)
        np.save(f"data/X_diagonal_n={n}_d={d}", X)

        # random singular
        X = np.random.rand(n, d)
        X = np.random.rand(n, int(3 * d / 4))
        X_last_column = X[:, -1].reshape(n, 1)
        X_added_columns = np.repeat(X_last_column, int(d / 4), axis=1)
        X = np.hstack((X, X_added_columns))
        np.save(f"data/X_singular_n={n}_d={d}", X)

        # add noise
        # sigma = 0.5
        # noise = np.random.normal(0, sigma, size=(X_added_columns.shape))
        # X = np.hstack((X, X_added_columns+noise))

        # output
        y = np.random.rand(n).reshape(n, 1)
        np.save(f"data/y_n={n}_d={d}", y)
