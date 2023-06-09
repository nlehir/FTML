"""
    Compute the similarity matrix
    to use for spectral clustering
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

if __name__ == "__main__":
    data_path = os.path.join("data", "data.npy")
    data = np.load(data_path)

    # compute pairwise distances
    distances = cdist(data, data)

    # we use the standard deviation to compute the similarity
    similarity = np.exp(-distances / distances.std())
    print(f"distances standard deviation: {distances.std()}")
    plt.imshow(similarity)
    fig_path = os.path.join("images", "similarity.pdf")
    plt.savefig(fig_path)
    plt.close()

    selection = np.where(similarity > 0.5)
    adjacency_matrix = np.zeros(distances.shape)
    adjacency_matrix[selection] = 1
    plt.imshow(adjacency_matrix)
    fig_path = os.path.join("images", "adjacency_matrix.pdf")
    plt.savefig(fig_path)
    plt.close()

    # save the adjacency matrix
    np.save(os.path.join("data", "adjacency_matrix"), adjacency_matrix)
