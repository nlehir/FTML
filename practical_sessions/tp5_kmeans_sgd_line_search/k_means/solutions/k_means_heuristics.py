"""
    Assess the quality of the clustering using the inertia knee criterion
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from seaborn import pairplot
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer


def main():
    # load the data
    data = np.load("data.npy")

    nbs_of_clusters = range(2, 15)
    inertias = list()
    silhouettes = list()

    for nb_of_clusters in nbs_of_clusters:
        kmeans = KMeans(n_clusters=nb_of_clusters, n_init="auto").fit(data)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(data, kmeans.labels_, metric="euclidean")
        inertias.append(inertia)
        silhouettes.append(silhouette)
        message = (
            f"{nb_of_clusters} clusters:\n"
            f"inertia = {inertia:.2E}\n"
            f"silhouette = {silhouette:.2E}\n\n"
        )
        print(message)

    # plot the inertia
    plt.plot(nbs_of_clusters, inertias)
    plt.xlabel("number of centroids")
    plt.xticks(range(1, 13))
    plt.ylabel("inertia")
    plt.savefig("inertia.pdf")
    plt.close()

    # plot the silhouette
    plt.plot(nbs_of_clusters, silhouettes)
    plt.xlabel("number of centroids")
    plt.xticks(range(1, 13))
    plt.ylabel("silhouette")
    plt.savefig("silhouette.pdf")
    plt.close()

    # scatter plot the data
    df = pd.DataFrame(data)
    pairplot(df)
    plt.savefig("scatter_matrix.pdf")
    plt.close()

    """
        Use an algorithm to find the knee (maximum curvature)
        https://github.com/arvkevi/kneed
    """
    kneedle = KneeLocator(
        nbs_of_clusters, inertias, S=1.0, curve="convex", direction="decreasing"
    )
    print(f"knee at {kneedle.knee} clusters\n")
    # print(kneedle.knee_y)

    """
        Use yellowbrick
    """
    metrics = ["distortion", "silhouette", "calinski_harabasz"]
    for metric in metrics:
        if metric == "calinski_harabasz":
            locate_elbow = False
        else:
            locate_elbow = True
        print(f"yellowbrick with {metric}")
        kmeans = KMeans(n_init="auto")
        visualizer = KElbowVisualizer(
            kmeans, metric=metric, locate_elbow=locate_elbow, k=(2, 10)
        )
        visualizer.fit(data)
        visualizer.finalize()
        plt.savefig(f"yellowbrick_{metric}.pdf")
        plt.close()


if __name__ == "__main__":
    main()