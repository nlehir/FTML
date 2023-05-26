"""
    Generate data to compare OLS and Ridge
"""
import numpy as np
import os
from constants import rng

def main():
    n_list = [30]
    d_list = [10, 20, 30]

    for n in n_list:
        for d in d_list:
            # to generate correlated columns (high variance of OLS)
            # sigma = 0.02
            # X = np.random.rand(n, d-1)
            # X_last_column = X[:, -1].reshape(n, 1)
            # noise = np.random.normal(0, sigma, size=(X_last_column.shape))
            # X_added_column = X_last_column + noise
            # X = np.hstack((X, X_added_column))
            X = rng.uniform(size=(n, d))
            path = os.path.join("data", f"n_design_matrix_n={n}_d={d}")
            np.save(path, X)

if __name__ == "__main__":
    main()
