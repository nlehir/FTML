"""
Observe a knn adaptivity: by adaptivity, we mean that
the estimator automatically benefits by essence from a specific structure in the data.

Adaptivity is a good property of estimators, and we will
discuss it more in the course.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from constants import k
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from utils_solutions import predict_knn


def knn(
    n_samples: int, x_data: np.ndarray, y_data: np.ndarray, x_data_test: np.ndarray
) -> float:
    """
    Compute the test error of the knn predictor
    for a given number of input samples.
    """
    print(f"\n{k} neighbors, {n_samples} samples")
    x_data_n = x_data[:n_samples]
    y_data_n = y_data[:n_samples]
    n_tests = x_data_test.shape[0]
    y_predictions = predict_knn(x_data_n, y_data_n, x_data_test)
    y_truth = np.linalg.norm(x_data_test, axis=1)

    mean_squared_error = np.linalg.norm(y_predictions - y_truth) ** 2 / n_tests
    print(f"mean squared error {mean_squared_error:.2E}")
    return mean_squared_error


def main() -> None:
    # load data
    folder = "data_knn"
    x_data = np.load(os.path.join(folder, "x_data.npy"))
    x_data_test = np.load(os.path.join(folder, "x_data_test.npy"))
    y_data = np.load(os.path.join(folder, "y_data.npy"))
    d = x_data.shape[1]

    # simulation parameters
    k = 2
    n_list = [1, 10, 100, 1000]
    n_list = [1, 10, 100, 1000, 10000]

    mean_squared_errors = list()
    for n_samples in n_list:
        mean_squared_errors.append(knn(n_samples, x_data, y_data, x_data_test))

    plt.plot(
        n_list, mean_squared_errors, "o", label="mean squared error of knn estimator"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("number of samples")
    plt.ylabel("Mean squared error")
    plt.title(f"knn, k={k}, d={d}")  # \ndata on a subspace of dimension {support_dim}")

    # find slope of error as a function of the number of samples
    log_x_n = np.log(n_list)
    log_mse = np.log(mean_squared_errors)

    def objective(x, slope, b):
        return slope * x + b

    popt, _ = curve_fit(objective, list(log_x_n), list(log_mse))
    slope, b = popt
    xx = np.linspace(min(n_list), max(n_list), num=10)
    yy = np.power(xx, 1 / slope)
    zz = np.power(xx, -1 / d)
    print(f"slope: {slope:2f}")
    plt.plot(xx, yy, label=f"y=x^(1/{slope:.2f})")
    plt.plot(xx, zz, label=f"y=x^(-1/d)")
    plt.legend(loc="best")
    plt.savefig(f"images_knn/x_test_knn_k={k}.pdf")
    plt.close()

    # PCA to check the intrinsic dimensionality of the data
    # load the sklearn estimator
    print("PCA")
    pca = PCA()
    pca.fit(x_data)

    # principal component obtained by the algorithm
    # print("components")
    # print(pca.components_)

    # variance carried by those axes
    print(f"\nexplained variance {pca.explained_variance_}")

    # variance ratio carried by those axes
    print(f"\nexplained variance ratio {pca.explained_variance_ratio_}")

    variance_ratio = pca.explained_variance_ratio_
    nb_components = 20
    plt.plot(
        range(1, nb_components + 1), np.cumsum(variance_ratio[:nb_components]), "o"
    )
    plt.xticks(range(1, nb_components + 1))
    plt.title("variance in the PCA projected data")
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    plt.savefig("images_knn/explained_variance.pdf")


if __name__ == "__main__":
    main()
