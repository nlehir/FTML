"""
Generate data that lie on a lower dimensional manifold
"""

import numpy as np
import os


def main() -> None:
    support_dim = 3
    n_samples = int(1e4)
    d = 30

    rng = np.random.default_rng()

    sub_basis = rng.uniform(0, 1, (support_dim, d))
    x_data_coordinates = rng.uniform(0, 1, (n_samples, support_dim))
    x_data = x_data_coordinates @ sub_basis
    y_data = np.linalg.norm(x_data, axis=1)
    x_data_test_coordinates = np.random.uniform(0, 1, (n_samples, support_dim))
    x_data_test = x_data_test_coordinates @ sub_basis

    folder = "data_knn"
    np.save(os.path.join(folder, "x_data"), x_data)
    np.save(os.path.join(folder, "y_data"), y_data)
    np.save(os.path.join(folder, "x_data_test"), x_data_test)

if __name__ == "__main__":
    main()
