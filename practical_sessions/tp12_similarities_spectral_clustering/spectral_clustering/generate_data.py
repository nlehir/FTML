"""
    Generate data for the normalized cut heuristic
"""

import os

import numpy as np

if __name__ == "__main__":
    scale = 1
    std_1 = scale * 0.1
    std_2 = scale * 0.2
    std_3 = scale * 1

    rng = np.random.default_rng()
    cluster_1 = rng.normal((1, 4), std_1, (10, 2))
    cluster_2 = rng.normal((2, 2), std_2, (20, 2))
    cluster_3 = rng.normal((-2, -2), std_3, (30, 2))

    data = np.concatenate((cluster_1, cluster_2))
    data = np.concatenate((data, cluster_3))
    np.random.shuffle(data)

    data_path = os.path.join("data", "data")
    np.save(data_path, data)
