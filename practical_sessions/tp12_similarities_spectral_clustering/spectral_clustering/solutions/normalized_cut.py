"""
    Use the normalized cut heursitic for obtaining a relevant number of clusters
    in a clustering situation.
    The clustering will be performed by a Spectral Clustering.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import SpectralClustering

# load the data
adjacency_matrix_path = os.path.join("data", "adjacency_matrix.npy")
adjacency_matrix = np.load(adjacency_matrix_path)
n = adjacency_matrix.shape[0]


def cluster_and_compute_normalized_cut(nb_clusters, adjacency_matrix):
    # input the similarity matrix manually
    sc = SpectralClustering(nb_clusters, affinity="precomputed")
    sc.fit_predict(adjacency_matrix)

    # compute the normalized cut
    normalized_cut = 0
    for cluster_index in range(nb_clusters):
        cluster = np.where(sc.labels_ == cluster_index)[0]

        # compute degree of the cluster
        cluster_adjacencies = adjacency_matrix[cluster]
        cluster_degree = cluster_adjacencies.sum()

        # compute the number of outside connections
        complementary = np.setdiff1d(range(n), cluster)
        cluster_cut = adjacency_matrix[cluster][:, complementary].sum()

        # apply definition of normalized cut
        normalized_cut += cluster_cut/cluster_degree

    print(f"normalized cut: {normalized_cut}")
    return normalized_cut


if __name__ == "__main__":
    normalized_cuts = list()
    max_nb_clusters = 10
    tried_nb_clusters = range(1, max_nb_clusters)

    for nb_clusters in tried_nb_clusters:
        print(f"======\nnb clusters: {nb_clusters}")
        normalized_cuts.append(
            cluster_and_compute_normalized_cut(nb_clusters, adjacency_matrix)
        )

    plt.plot(tried_nb_clusters, normalized_cuts, "o")
    plt.title("normalized cut heuristic")
    plt.xlabel("nb clusters")
    plt.ylabel("normalized cut")
    fig_path = os.path.join("images", "normalized_cuts.pdf")
    plt.savefig(fig_path)
