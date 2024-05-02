import matplotlib.pyplot as plt
import os
import numpy as np


def plot_clustering(
    iteration: int,
    x: np.ndarray,
    y: np.ndarray,
    cluster_0: np.ndarray,
    cluster_1: np.ndarray,
    cluster_2: np.ndarray,
    x_centroids,
    y_centroids,
    step: str,
) -> None:
    """
    Plot the current state of the clustering

    """
    plt.plot(
        x[cluster_0], y[cluster_0], "o", color="darkorange", markersize=3, alpha=0.8
    )
    plt.plot(
        x[cluster_1], y[cluster_1], "o", color="firebrick", markersize=3, alpha=0.8
    )
    plt.plot(
        x[cluster_2], y[cluster_2], "o", color="cornflowerblue", markersize=3, alpha=0.8
    )
    plt.plot(x_centroids, y_centroids, "o", color="lime")
    title = f"update centroids : iteration {iteration} (centroids in green)"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    figpath = os.path.join("clusterings", f"it_{iteration}_{step}.pdf")
    plt.savefig(figpath)
    plt.close("all")


def clean(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))
