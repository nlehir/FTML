"""
Goal: obsevrve curse of dimensionality for local averaging
with nearest neighbors algorithm, on a dataset uniformly
sampled in the input space R^d.

The target function to predict is the euclidean norm on R^d,
but you may also try other functions.

If epsilon is the order of magnitude of the approximation error,
we have epsilon=O(n^{-1/d})

hence, log epsilon = O( -1/d * log n )

The function knn_d_n() and predict() should be edited.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from constants import d_list, k, n_samples_list, n_test, rng
from scipy.spatial.distance import cdist


def predict(x_data: np.ndarray, y_data: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """
    Predict with knn estimation

    Parameters:
        x_data (float matrix): (n_samples, d) samples in input space
        y_data (float vector): (n_samples, 1) values of the target function (labels)
        (here, it is the euclidean norm, for these samples)
        x_test (float matrix): (n_test, d) data for which we
        predict a value based on the dataset.

    Returns:
        y_predictions (float matrix): predictions for the data
        in x_test.
        y_predictions must be of shape (n_test,)

    The samples are called "data" instead of "train" because there
    is no training involved here !

    You need to edit this function.
    You can use cdist from scipy.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    """
    return 1


def knn_d_n(n_samples: int, d: int) -> tuple[float, float]:
    """
    Run knn simulation for (n, d) fixed by generating samples
    and computing the test error of a knn predictor.

    Parameters:
        n_samples: number of samples in dataset
        d: dimension of the input space

    Returns:
        mean_squared_test_error, maximum_test_error

    Edit this function.
    """
    # generate a random dataset
    # uniformly sampled un the input space

    # test

    return 1, 1


def knn_d(d, ax_mean, ax_max) -> None:
    """
    For some fixed dimension d,
    run knn for several number of samples in the dataset,
    in order to monitor the evolution of the test error as a function of
    the number of samples.

    Both the mean test error and the maximum test error are plotted.

    Parameters:
        d (integer): dimension of the input space
        ax_mean, ax_max : python subplots used
        to show the results.
    """
    mean_squared_errors = list()
    max_errors = list()
    print(f"\n------\nd={d}\n------")
    # iterate over the number of samples in the dataset
    for n_samples in n_samples_list:
        mean_squared_error, max_error = knn_d_n(n_samples=n_samples, d=d)
        mean_squared_errors.append(mean_squared_error)
        max_errors.append(max_error)
    ax_mean.plot(n_samples_list, mean_squared_errors, "o", label=f"d={d}")
    ax_max.plot(n_samples_list, max_errors, "o", label=f"d={d}")


def main() -> None:
    """
    Study the prediction error of the knn estimator
    for several input space dimensions d.
    """
    print(f"study kNN with {k} nearest neighbors")
    fig, (ax_mean, ax_max) = plt.subplots(2, figsize=[10, 10])

    # iterate over the input space dimension d
    for d in d_list:
        knn_d(d=d, ax_mean=ax_mean, ax_max=ax_max)

    # plot the results and set the plot
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
    fig_name = f"knn_k={k}_ntest={n_test}.pdf"
    fig_path = os.path.join(
        "images",
        fig_name,
    )
    fig.savefig(fig_path)


if __name__ == "__main__":
    main()
