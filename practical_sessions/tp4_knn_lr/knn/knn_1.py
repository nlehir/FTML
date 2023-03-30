"""
    Goal: obsevrve curse of dimensionality for local averaging
    with nearest neighbors algorithm.

    we have epsilon=O(n^{-1/d})

    hence, log epsilon = O( -1/d * log n )
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from constants import k, n_test, n_samples_list, rng, d_list


def predict(x_data: np.ndarray, y_data: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """
        Predict with knn estimation

        Parameters:
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
    return 1


def knn_d_n(n_samples: int, d: int) -> tuple[float, float]:
    """
        Run knn simulation for (n, d) fixed.

        Parameters:
            n_samples: number of samples in dataset
            d: dimension of the input space
    """
    # generate dataset

    # test

    return 1, 1


def knn_d(d, ax_mean, ax_max) -> None:
    """
        For some fixed dimension d,
        run knn for several number of samples in the dataset,
        in order to monitor the evolution of the error as a function of
        the number of samples.

        Parameters:
            d (integer): dimension of the input space
            ax_mean, ax_max : python subplots used
            to show the results.
    """
    mean_squared_errors = list()
    max_errors = list()
    print(f"\n------\nd={d}\n------")
    for n_samples in n_samples_list:
        mean_squared_error, max_error = knn_d_n(n_samples, d)
        mean_squared_errors.append(mean_squared_error)
        max_errors.append(max_error)
    ax_mean.plot(n_samples_list, mean_squared_errors, "o", label=f"d={d}")
    ax_max.plot(n_samples_list, max_errors, "o", label=f"d={d}")


def main() -> None:
    print(f"study kNN with {k} samples")
    fig, (ax_mean, ax_max) = plt.subplots(2, figsize=[10, 10])
    for d in d_list:
        knn_d(d, ax_mean, ax_max)


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


if __name__ == "__main__":
    main()
