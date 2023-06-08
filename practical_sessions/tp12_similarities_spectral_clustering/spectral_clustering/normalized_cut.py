"""
    We will study a heursitic for obtaining a relevant number of clusters
    in a clustering situation.
    The clustering will be performed by a Spectral Clustering.
    Spectral Clustering works with an adjacency matrix
    or a similarity matrix.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering

# load the data
data_path = os.path.join("data", "data.npy")
data = np.load(data_path)
plt.scatter(data[:, 0], data[:, 1])
fig_path = os.path.join("images", "data_to_cluster.pdf")
plt.savefig(fig_path)
