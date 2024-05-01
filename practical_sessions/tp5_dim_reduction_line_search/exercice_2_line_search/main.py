"""
    Gradient descent (GD) on a strongly convex
    loss function.
    The design matrix is randomly generated.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from algorithms import (
        gradient_descent,
        line_search,
        OLS_estimator,
        )

GAMMAS = [0.1, 0.01, 0.001]
NUMBER_OF_ITERATIONS = 10000
N = 60
D = 40

def load_data(n, d):
    """
    Several data dimensions and dataset sizes are available in
    data/
    """
    folder = "./data"
    X_path = os.path.join(folder, f"X_gaussian_n={n}_d={d}.npy")
    y_path = os.path.join(folder, f"y_n={n}_d={d}.npy")
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y


def main() -> None:
    """
    Load the data
    """
    X, y = load_data(n=N, d=D)

    """
    Compute the OLS estimator and the Hessian
    """
    theta_star = OLS_estimator(X, y)
    H = 1 / N * np.transpose(X) @ X

    fig, (ax_linear, ax_log) = plt.subplots(nrows=1, ncols=2)
    for gamma in GAMMAS:
        gradient_descent(
                X=X,
                H=H,
                y=y,
                theta_star=theta_star,
                gamma=gamma,
                n_iterations=NUMBER_OF_ITERATIONS,
                ax_linear=ax_linear,
                ax_log=ax_log,
                )

    # line_search(
    #         X=X,
    #         H=H,
    #         y=y,
    #         theta_star=theta_star,
    #         n_iterations=NUMBER_OF_ITERATIONS,
    #         ax_linear=ax_linear,
    #         ax_log=ax_log,
    #         )
    title = (
        'Gradient descent: squared distance to the OLS estimator\n'
        r"$||\theta-\theta^*||^2$"
            )
    fig.suptitle(title)
    ax_linear.set_title("Linear scale")
    ax_linear.set_ylabel(r"$||\theta-\theta_{*}||^2$")
    ax_log.set_ylabel(r"$||\theta-\theta_{*}||^2$")
    ax_linear.set_xlabel("GD iteration")
    ax_log.set_xlabel("GD iteration")
    ax_log.set_title("Semilog scale")
    ax_log.set_yscale("log")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("Gradient_descent.pdf")


if __name__ == "__main__":
    main()
