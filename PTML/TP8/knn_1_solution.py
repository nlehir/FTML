"""
    Goal: obsevrve curse of dimensionality for local averaging
    with nearest neighbors algorithm.

    we have epsilon=O(n^{-1/d})

    hence, log epsilon = O( -1/d * log n )
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

n_test = 1000
sigma = 0
np.random.seed(13)


def predict(x_data, y_data, x_test, k):
    """
        Predict with knn estimation

        Parameters:
            k (integer): number of nearest neighbors used.
            x_data (float matrix): (n_samples, d) samples in input space
            y_data (float vector): (n_samples, 1) values of the target function
            (here, it is the euclidean norm, for these samples)
            x_test (float matrix): (n_samples, d) data for which we
            predict a value based on the dataset.

        Returns:
            y_predictions (float matrix): predictions for the data
            in x_test.
            y_predictions must be of shape (n_test,)

        You need to edit this function.
        You can use cdist from scipy.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    """
    dist_x_data = cdist(x_test, x_data, "euclidean")
    sorted_indexes = np.argsort(dist_x_data, axis=1)
    k_neighbors = sorted_indexes[:, :k]
    y_predictions = np.zeros(n_test)
    for i in range(len(x_test)):
        x_neighbors = k_neighbors[i]
        y_neighbors = y_data[x_neighbors]
        y_prediction = np.mean(y_neighbors)
        y_predictions[i] = y_prediction
    return y_predictions


def knn(n_samples, k, d):
    """
        Run knn with a choice parameters.

        Parameters:
            n_samples: number of samples in dataset
            k (integer): number of nearest neighbors
            d (integer): dimension of the input space
    """
    # generate dataset
    x_data = np.random.uniform(0, 1, (n_samples, d))
    y_data = np.linalg.norm(x_data, axis=1)

    # test
    x_test = np.random.uniform(0, 1, (n_test, d))
    y_predictions = predict(x_data, y_data, x_test, k)
    y_truth = np.linalg.norm(x_test, axis=1)

    # error
    mean_squared_error = np.linalg.norm(y_predictions - y_truth)**2/n_test
    max_error = np.amax(np.max(y_predictions - y_truth))

    print(f"\n{k} neighbors, {n_samples} samples")
    print(f"mean squared error {mean_squared_error:.2E}")
    return mean_squared_error, max_error


def knn_d(d, k, ax_mean, ax_max):
    """
        For some fixed dimension d,
        run knn for several number of samples in the dataset,
        in order to monitor the evolution of the error as a function of
        the number of samples.

        Parameters:
            d (integer): dimension of the input space
            k (integer): number of nearest neighbors
            ax_mean, ax_max : python subplots in order
            to show the results.
    """
    mean_squared_errores = list()
    max_errores = list()
    print(f"\n------\nd={d}\n------")
    for n_samples in n_samples_list:
        mean_squared_error, max_error = knn(n_samples, k, d)
        mean_squared_errores.append(mean_squared_error)
        max_errores.append(max_error)
    ax_mean.plot(n_samples_list, mean_squared_errores, "o", label=f"d={d}")
    ax_max.plot(n_samples_list, max_errores, "o", label=f"d={d}")


k=2
fig, (ax_mean, ax_max) = plt.subplots(2, figsize=[10, 10])
n_samples_list = [10, 100, 1000, 10000, 100000]
for d in [10, 100, 1000]:
    knn_d(d, k, ax_mean, ax_max)


ax_mean.set_ylabel("mean squared error")
ax_mean.set_xscale("log")
ax_mean.set_yscale("log")
ax_mean.legend(loc="best")
ax_max.set_ylabel("maximum error")
ax_max.set_xscale("log")
ax_max.set_yscale("log")
ax_max.set_xlabel("number of samples")

fig.suptitle(f"knn, k={k}")
fig.set_tight_layout(True)
fig.savefig(f"images_knn/knn_k={k}_ntest={n_test}.pdf")
