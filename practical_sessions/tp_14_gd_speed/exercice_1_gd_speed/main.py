"""
    Gradient descent (GD) on a strongly convex
    loss function.
    The design matrix is randomly generated.

    Compare the GD speed with the theoretical bound found for strongly convex
    functions.

"""
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np

from algorithms_solutions import (
        gradient_descent,
        upper_bound,
        # line_search,
        OLS_estimator,
        )
from params import N, D, MATRIX_TYPE

GAMMAS = [0.01]

NUMBER_OF_ITERATIONS = 10000
TOL = 1e-7
FONTSIZE = 10
COLORS = [
        "blue",
        "orange",
        "green",
        "red",
        "magenta",
        ]

def load_data(n, d, matrix_type):
    """
    Several data dimensions and dataset sizes are available in
    data/

    If the selected data do not already exist, you can generate them by running
    generate_data.py
    """
    folder = "./data"
    X_path = os.path.join(folder, f"X_{matrix_type}_n={n}_d={d}.npy")
    y_path = os.path.join(folder, f"y_n={n}_d={d}.npy")
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y


def main() -> None:
    """
    Load the data
    """
    X, y = load_data(n=N, d=D, matrix_type=MATRIX_TYPE)


    """
    Compute the OLS estimator and the Hessian
    """
    tic  =  time()
    theta_hat = OLS_estimator(X, y)
    toc  =  time()
    ols_time = toc - tic
    print(f"OLS time {ols_time:1E}")
    H = 1 / N * np.transpose(X) @ X
    eigenvalues, _ = np.linalg.eig(H)
    L = eigenvalues.max()
    GAMMAS.append(1/L)

    fig, (ax_linear, ax_log) = plt.subplots(nrows=1, ncols=2, figsize=(14, 10))
    color_index = 0
    _, d = X.shape
    theta_0 = np.zeros((d, 1))
    for index, gamma in enumerate(GAMMAS):
        print(f"\nGradient descent with gamma {gamma}")
        gamma_from_H = True if index == len(GAMMAS)-1 else False
        gradient_descent(
                X=X,
                H=H,
                y=y,
                theta_hat=theta_hat,
                gamma=gamma,
                n_iterations=NUMBER_OF_ITERATIONS,
                ax_linear=ax_linear,
                ax_log=ax_log,
                tol=TOL,
                color=COLORS[color_index],
                theta_0=theta_0,
                gamma_from_H=gamma_from_H,
                )
        color_index += 1

    upper_bounds = upper_bound(
            theta_0=theta_0,
            theta_hat=theta_hat,
            H=H,
            n_iterations=NUMBER_OF_ITERATIONS,
            )
    label = (
            "upper bound "
            r"$\exp(\frac{-2t}{\kappa})$"
            )
    ax_linear.plot(upper_bounds, label=label)
    ax_log.plot(upper_bounds, label=label)

    # print(f"\nLine search")
    # line_search(
    #         X=X,
    #         H=H,
    #         y=y,
    #         theta_hat=theta_hat,
    #         n_iterations=NUMBER_OF_ITERATIONS,
    #         ax_linear=ax_linear,
    #         ax_log=ax_log,
    #         tol=TOL,
    #         color=COLORS[color_index],
    #         )
    title = (
        'Gradient descent: squared distance to the OLS estimator\n'
        r"$||\theta-\hat{\theta}||^2$"
        f"\nn={N}, d={D}\n"
        f"{MATRIX_TYPE} matrix\n"
        f"OLS time: {ols_time:.1E}"
            )
    fig.suptitle(title)
    ax_linear.set_title("Linear scale")
    ax_linear.set_ylabel(r"$||\theta-\hat{\theta}||^2$")
    ax_log.set_ylabel(r"$||\theta-\hat{\theta}||^2$")
    ax_linear.set_xlabel("GD iteration")
    ax_log.set_xlabel("GD iteration")
    ax_log.set_title("Log scale")
    ax_log.set_yscale("log")
    ax_log.set_xscale("log")
    plt.legend(loc="best", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"Gradient_descent_n_{N}_d_{D}_{MATRIX_TYPE}.pdf")


if __name__ == "__main__":
    main()
