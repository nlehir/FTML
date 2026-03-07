"""
Find the optimal parameters for a 1D linear regression
and plot the prediction made by this estimator.
"""

import os

import matplotlib.pyplot as plt

# from utils_solution import empirical_risk
from constants import STD_NOISE
from utils import compute_optimal_params, empirical_risk
from utils_files import clean_filename, load_data


def main():
    print(f"Optimal linear regression, standard deviation {STD_NOISE}")

    X_train, X_test, y_train, y_test = load_data(std=STD_NOISE)

    best_theta, best_b = compute_optimal_params(X=X_train, y=y_train)
    train_error = empirical_risk(
        theta=best_theta,
        b=best_b,
        X=X_train,
        y=y_train,
    )
    test_error = empirical_risk(
        theta=best_theta,
        b=best_b,
        X=X_test,
        y=y_test,
    )
    print(
        "best theta:"
        f" {best_theta:.2f}"
        "\nbest b:"
        f" {best_b:.2f}"
        "\ntrain_error:"
        f" {train_error:.2f}"
        "\ntest_error:"
        f" {test_error:.2f}"
    )

    # plot dataset
    plt.plot(X_train, y_train, "o", alpha=0.7, label="train")
    plt.plot(X_test, y_test, "o", alpha=0.7, label="test")

    """
    Compute and plot the predictions of the linear regressor
    """

    # save figure
    plt.xlabel("temperature (°C)")
    plt.ylabel("power_consumption (MW)")
    plt.legend(loc="best")
    title = (
        "Linear regression"
        "\n(empirical risk minimization)"
        f"\ntrain error: {train_error:.1f}"
        f"\ntest error: {test_error:.1f}"
    )
    plt.title(title)
    plt.tight_layout()
    fig_name = f"optimal_linear_regression_standard_deviation_{STD_NOISE}"
    fig_name = f"{clean_filename(fig_name)}.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)


if __name__ == "__main__":
    main()
