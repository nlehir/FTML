"""
    Goal: obsevrve curse of dimensionality for local averaging
    with nearest neighbors algorithm, on a dataset uniformly
    sampled in the input space R^d.

    The target function to predict is the euclidean norm on R^d,
    but you may also try other functions.

    If epsilon is the order of magnitude of the approximation error,
    we have epsilon=O(n^{-1/d})

    hence, log epsilon = O( -1/d * log n )

    The function knn_d_n() and predict_knn() in utils.py should be edited.
"""
import numpy as np
import matplotlib.pyplot as plt
from constants import k, n_test, n_samples_list, rng
from utils_solutions import predict_knn


def knn_d_n(n_samples: int, d: int) -> tuple[float, float]:
    """
        Run knn simulation for (n, d) fixed by generating samples
        and computing the test error of a knn predictor.

        Parameters:
            n_samples: number of samples in dataset
            d: dimension of the input space

        Returns:
            mean_squared_test_error, maximum_test_error
    """
    # generate dataset
    # uniformly sampled un the input space
    x_data = rng.uniform(low=0, high=1, size=(n_samples, d))
    y_data = np.linalg.norm(x=x_data, axis=1)

    # test
    x_test = rng.uniform(0, 1, (n_test, d))
    y_predictions = predict_knn(
            x_data=x_data,
            y_data=y_data,
            x_test=x_test,
            )
    y_truth = np.linalg.norm(x_test, axis=1)

    # error
    mean_squared_error = np.linalg.norm(y_predictions - y_truth)**2/n_test
    max_error = np.amax(np.max(y_predictions - y_truth))

    print(f"\n{n_samples} samples")
    print(f"MSE: {mean_squared_error:.2E}")
    return mean_squared_error, max_error


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
    for n_samples in n_samples_list:
        mean_squared_error, max_error = knn_d_n(n_samples, d)
        mean_squared_errors.append(mean_squared_error)
        max_errors.append(max_error)
    ax_mean.plot(n_samples_list, mean_squared_errors, "o", label=f"d={d}")
    ax_max.plot(n_samples_list, max_errors, "o", label=f"d={d}")


def main() -> None:
    """
    Study the prediction error of the knn estimator
    for several input space dimensions d.
    """
    print(f"study kNN with {k} samples")
    fig, (ax_mean, ax_max) = plt.subplots(2, figsize=[10, 10])
    for d in [10, 100, 1000]:
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
