"""
    Gradient descent (GD) on a strongly convex
    loss function.
    The design matrix is randomly generated.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from algorithms import OLS_estimator, gradient
from optimal_gamma import compute_gamma_star


def main() -> None:
    """
    Load the data
    """
    n = 60
    d = 40
    folder = "./data"
    X_path = os.path.join(folder, f"X_gaussian_n={n}_d={d}.npy")
    y_path = os.path.join(folder, f"y_n={n}_d={d}.npy")
    X = np.load(X_path)
    y = np.load(y_path)

    """
        Compute the important quantities
    """
    # Hessian matrix
    H = 1 / n * np.matmul(np.transpose(X), X)
    # compute spectrum of H
    eigenvalues, eigenvectors = np.linalg.eig(H)
    # sort the eigenvalues
    sorted_indexes = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvectors = eigenvectors[sorted_indexes]
    # compute strong convexity and smoothness
    L = eigenvalues[0]
    mu = eigenvalues[-1]
    kappa = L / mu
    print(f"L: {L}")
    print(f"mu: {mu}")
    print(f"kappa: {kappa}")
    # OLS estimator
    eta_star = OLS_estimator(X, y)

    """
        Preparation of the algorithms
    """
    theta_0 = np.zeros((d, 1))
    number_of_iterations = 5000
    gamma_gd = 1 / L
    GD_distances_to_opt = list()
    LS_distances_to_opt = list()

    theta_GD = theta_0.copy()
    theta_LS = theta_0.copy()

    """
        Algorithm
        Run constant step-size GD
        and GD Line seach in parallel
    """
    for _ in range(1, number_of_iterations + 1):
        GD_distances_to_opt.append(np.linalg.norm(theta_GD - eta_star) ** 2)
        LS_distances_to_opt.append(np.linalg.norm(theta_LS - eta_star) ** 2)

        # GD update
        theta_GD -= gamma_gd * gradient(theta_GD, H, X, y)

        # GD Line search update
        grad = gradient(theta_LS, H, X, y)
        gamma_star = compute_gamma_star(H, grad)
        theta_LS -= gamma_star * gradient(theta_LS, H, X, y)

    """
        Plot the results
    """
    # plot logarithm of the distance to optimal estimator
    plt.plot(
        range(1, number_of_iterations + 1), np.log10(GD_distances_to_opt), label="GD"
    )
    plt.plot(
        range(1, number_of_iterations + 1),
        np.log10(LS_distances_to_opt),
        label="Line search",
    )
    plt.xlabel("iteration")
    plt.ylabel(r"$\log_{10}(||\theta-\eta_{*}||^2)$")
    plt.ylabel(r"$||\theta-\eta_{*}||^2$")
    plt.title(
        "Constant step-size gradient descent vs exact line search\n"
        + r"$||\theta-\eta_{*}||^2$"
    )
    plt.legend(loc="best", prop={"size": 9})
    plt.tight_layout()
    figpath  = os.path.join("images", "LS_strongly_convex_semilog.pdf")
    plt.savefig(figpath)
    plt.close()

    # plot the distance to optimal estimator
    plt.plot(range(1, number_of_iterations + 1), GD_distances_to_opt, label="GD")
    plt.plot(
        range(1, number_of_iterations + 1), LS_distances_to_opt, label="Line search"
    )
    plt.xlabel("iteration")
    plt.ylabel(r"$||\theta-\eta_{*}||^2$")
    plt.title(
        "Constant step-size gradient descent vs exact line search\n"
        + r"$||\theta-\eta_{*}||^2$"
    )
    plt.legend(loc="best", prop={"size": 9})
    plt.tight_layout()
    figpath  = os.path.join("images", "LS_strongly_convex.pdf")
    plt.savefig(figpath)

if __name__ == "__main__":
    main()
