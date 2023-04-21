"""
    heavy-ball on a strongly convex least-squares loss funtion
    The design matrix was randomly generated.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    gradient,
    error,
    upper_bound_strongly_convex,
    generate_output_data,
)


def main() -> None:
    # load the data
    data_folder = "data"
    X_path = os.path.join(data_folder, "X.npy")
    X = np.load(X_path)
    rank = np.linalg.matrix_rank(X)
    n, d = X.shape
    print(f"n: {n}")
    print(f"d: {d}")
    print(f"rank of X: {rank}")

    # generate output data
    sigma = 0
    r = np.random.default_rng()
    theta_star = r.uniform(-1, 1, size=(d, 1))
    y = generate_output_data(X, theta_star, sigma, r)


if __name__ == "__main__":
    main()
