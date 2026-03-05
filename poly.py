"""
Find the optimal parameters for a 1D linear regression
and plot the prediction made by this estimator.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from constants import MEAN_NOISE, STD_NOISE
from utils import compute_optimal_params, empirical_risk
from utils_files import clean_filename, load_data
from utils_solution import empirical_risk


def fit_polynom(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    degree: int,
):
    print(f"fit polynom of degree {degree}")
    """
    Fit a polynom of the degree to the data
    """
    polynom = np.polyfit(x=X_train, y=y_train, deg=degree)

    train_predictions = np.polyval(p=polynom, x=X_train)
    test_predictions = np.polyval(p=polynom, x=X_test)
    X_plot = np.linspace(X_train.min(), X_train.max(), num=200)
    y_plot = np.polyval(p=polynom, x=X_plot)

    """
    Compute the train and test error
    """
    n_train = X_train.shape[0]
    train_error = (np.linalg.norm(train_predictions - y_train) ** 2) / n_train
    n_test = X_test.shape[0]
    test_error = (np.linalg.norm(test_predictions - y_test) ** 2) / n_test

    """
    Plot
    """
    plt.plot(X_train, y_train, "o", alpha=0.7, label="train")
    plt.plot(X_test, y_test, "o", alpha=0.7, label="test")
    plt.plot(X_plot, y_plot, alpha=0.7, label="fitted polynom")

    plt.xlabel("temperature (°C)")
    plt.ylabel("power_consumption (MW)")
    plt.legend(loc="best")
    title = (
        f"Polynomial regression, degree {degree}"
        "\n(empirical risk minimization)"
        f"\ntrain error: {train_error:.1f}"
        f"\ntest error: {test_error:.1f}"
        f"\nnumber of train samples: {n_train}"
        f"\nnoise standard deviation: {STD_NOISE:.1f}"
    )
    plt.title(title)
    plt.tight_layout()
    file_name = f"polyomial_regression_std_{STD_NOISE:.1f}_degree_{degree}"
    file_name = clean_filename(name=file_name)
    fig_path = os.path.join("images", "polynomial_regression", f"{file_name}.pdf")
    plt.savefig(fig_path)
    plt.close()
    return train_error, test_error


def main():
    print(f"Polynomial regression, standard deviation {STD_NOISE}")
    X_train, X_test, y_train, y_test = load_data(std=STD_NOISE)

    degrees = range(2, 40)
    train_errors = list()
    test_errors = list()
    for degree in degrees:
        train_error, test_error = fit_polynom(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            degree=degree,
        )
        train_errors.append(train_error)
        test_errors.append(test_error)

    # plot dataset
    plt.plot(train_errors, "o", alpha=0.7, label="train error")
    plt.plot(test_errors, "o", alpha=0.7, label="test error")
    plt.xlabel("degree")
    plt.ylabel("squared error")
    plt.yscale("log")
    plt.legend(loc="best")
    title = f"Polynomial regression\nnoise standard deviation {STD_NOISE}"
    plt.title(title)
    plt.tight_layout()
    fig_name = f"polynomial_regression_standard_deviation_{STD_NOISE}"
    fig_name = f"{clean_filename(fig_name)}.pdf"
    fig_path = os.path.join("images", "polynomial_regression", fig_name)
    plt.savefig(fig_path)


if __name__ == "__main__":
    main()
