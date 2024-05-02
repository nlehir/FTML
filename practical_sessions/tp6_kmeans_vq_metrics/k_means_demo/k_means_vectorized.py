"""
    Perform the k-means algorithm (unsupervised learning)
"""
import os

import matplotlib.pyplot as plt
import numpy as np


def main(nbs_of_iterations: int) -> None:
    """
    perform a k-means algorithm
    with 3 clusters on a simple 2D dataset.
    """

    # load the data
    datapath = os.path.join("data", "data.npy")
    data = np.load(datapath)
    x = data[:, 0]
    y = data[:, 1]

    nb_samples = len(x)

    # clean images
    clean("images")

    x_min = min(x)
    x_max = max(x)

    rng = np.random.default_rng()
    centroids = rng.uniform(low=x_min, high=x_max, size=(3, 2))
    print("initial centroid positions")
    print(f"centroids: {centroids}")


    for iteration in range(nbs_of_iterations):
        print(f"\niteration: {iteration}")

        diff_0 = data-centroids[0]
        diff_1 = data-centroids[1]
        diff_2 = data-centroids[2]

        dist_0 = np.linalg.norm(diff_0, axis=1)
        dist_1 = np.linalg.norm(diff_1, axis=1)
        dist_2 = np.linalg.norm(diff_2, axis=1)

        dists = np.vstack((dist_0, dist_1, dist_2))
        centroids_assignments = np.argmin(dists, axis=0)

        cluster_0 = np.where(centroids_assignments == 0)[0]
        cluster_1 = np.where(centroids_assignments == 1)[0]
        cluster_2 = np.where(centroids_assignments == 2)[0]

        x_centroids = centroids[:, 0]
        y_centroids = centroids[:, 1]
        plot_clustering(
            iteration,
            x,
            y,
            cluster_0,
            cluster_1,
            cluster_2,
            x_centroids,
            y_centroids,
            "assign_samples_to_centroid",
        )

        # Update centroids positions
        centroids[0] = np.mean(data[cluster_0], axis=0)
        centroids[1] = np.mean(data[cluster_1], axis=0)
        centroids[2] = np.mean(data[cluster_2], axis=0)
        print(f"x0: {x_centroids[0]:.2f}  y0: {y_centroids[0]:.2f}")
        print(f"x1: {x_centroids[1]:.2f}  y1: {y_centroids[1]:.2f}")
        print(f"x2: {x_centroids[2]:.2f}  y2: {y_centroids[2]:.2f}")

        plot_clustering(
            iteration,
            x,
            y,
            cluster_0,
            cluster_1,
            cluster_2,
            x_centroids,
            y_centroids,
            "move_centroids",
        )


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
    parsed_step = step.replace("_", " ")
    title = f"{parsed_step} : iteration {iteration} (centroids in green)"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    figpath = os.path.join("images", f"it_{iteration}_{step}.pdf")
    plt.savefig(figpath)
    plt.close("all")


def clean(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))


if __name__ == "__main__":
    nbs_of_iterations = 10
    main(nbs_of_iterations)
