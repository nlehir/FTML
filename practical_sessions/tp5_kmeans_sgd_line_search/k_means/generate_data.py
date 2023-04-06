"""
    Generate the data used in exercise 3
"""

import numpy as np
from scipy.spatial.distance import cdist

rng = np.random.default_rng()

stds = 50

mean_1 = (600, 2000, 2000, 20)
std_1 = stds

mean_2 = (200, -800, 45, -700)
std_2 = stds

mean_3 = (-400, -500, -100, -2000)
std_3 = stds

mean_4 = (-800, 300, -600, -400)
std_4 = stds

mean_5 = (1000, 900, -1000, 500)
std_5 = stds

mean_6 = (2100, 15, 800, 1000)
std_6 = stds

centers_array = np.asarray((mean_1, mean_2, mean_3, mean_4, mean_5, mean_6))
centers_dists = cdist(centers_array, centers_array)

n_cluster = 300

# generate the data
data_1 = rng.normal(loc=mean_1, scale=std_1, size=(n_cluster, 4))
data_2 = rng.normal(loc=mean_2, scale=std_2, size=(n_cluster, 4))
data_3 = rng.normal(loc=mean_3, scale=std_3, size=(n_cluster, 4))
data_4 = rng.normal(loc=mean_4, scale=std_4, size=(n_cluster, 4))
data_5 = rng.normal(loc=mean_5, scale=std_5, size=(n_cluster, 4))
data_6 = rng.normal(loc=mean_6, scale=std_6, size=(n_cluster, 4))
data = np.concatenate((data_1, data_2))
data = np.concatenate((data, data_3))
data = np.concatenate((data, data_4))
data = np.concatenate((data, data_5))
data = np.concatenate((data, data_6))

np.save("data.npy", data)
