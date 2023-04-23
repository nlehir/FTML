"""
    Comparison between Heavy-ball (HB) and
    gradient descent (GD) on a strongly convex
    loss function.
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import generate_output_data, gradient, upper_bound_strongly_convex


def main():
    """
    Load the data
    """
    data_folder = "data"
    X_path = os.path.join(data_folder, "X.npy")
    X = np.load(X_path)
    rank = np.linalg.matrix_rank(X)
    n, d = X.shape
    print(f"n: {n}")
    print(f"d: {d}")
    print(f"rank of X: {rank}")

    # generate output data
    sigma = 0
    r = np.random.default_rng()
    theta_star = r.uniform(-1, 1, size=(d, 1))
    y = generate_output_data(X, theta_star, sigma, r)

    """
        Compute the relevant quantities
    """
    # Hessian matrix
    H = 1 / n * np.matmul(np.transpose(X), X)
    # compute spectrum of H
    eigenvalues, eigenvectors = np.linalg.eig(H)
    # sort the eigenvalues
    sorted_indexes = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvalues = np.real(eigenvalues)
    eigenvectors = eigenvectors[sorted_indexes]
    # compute convexity and smoothness
    L = eigenvalues[0]
    mu = eigenvalues[-1]
    kappa = L / mu
    print(f"L: {L}")
    print(f"mu: {mu}")
    print(f"kappa: {kappa}")

    """
        Preparation of the algorithm
        See the exercises for a justification of the values used.
    """
    theta_0 = np.zeros((d, 1))
    number_of_iterations = 10000
    gamma_HB = 4 / (math.sqrt(L) + math.sqrt(mu)) ** 2
    gamma_GD = 1 / L
    beta = ((math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))) ** 2
    distances_to_opt_GD = list()
    distances_to_opt_HB = list()
    upper_bounds_GD = list()
    upper_bounds_HB = list()
    iteration_range = range(1, number_of_iterations)

    """
        Algorithm
    """
    # perform one update
    theta_GD = theta_0.copy()
    theta_HB = theta_0.copy()
    theta_HB_before = theta_0.copy()
    theta_HB -= gamma_HB * gradient(theta_HB, H, X, y)
    for iteration in iteration_range:
        distances_to_opt_HB.append(np.linalg.norm(theta_HB - theta_star))
        distances_to_opt_GD.append(np.linalg.norm(theta_GD - theta_star))
        upper_bounds_GD.append(
            upper_bound_strongly_convex(iteration, kappa, theta_0, theta_star)
        )
        upper_bounds_HB.append(
            upper_bound_strongly_convex(
                iteration, math.sqrt(kappa), theta_0, theta_star
            )
        )
        # HB update
        theta_HB_backup = theta_HB.copy()
        theta_HB = (
            theta_HB
            - gamma_HB * gradient(theta_HB, H, X, y)
            + beta * (theta_HB - theta_HB_before)
        )
        theta_HB_before = theta_HB_backup
        # GD update
        theta_GD -= gamma_GD * gradient(theta_GD, H, X, y)

    """
        Plot the results
    """
    plot_range = iteration_range
    plt.plot(np.log10(plot_range), np.log10(distances_to_opt_GD), label="GD")
    plt.plot(np.log10(plot_range), np.log10(distances_to_opt_HB), label="Heavy Ball")
    plt.plot(
        np.log10(plot_range),
        np.log10(upper_bounds_GD),
        label="upper bound, strongly convex loss function: "
        + r"$-\frac{2t}{\kappa}+\log_{10}(||\theta_0-\eta_{*}||^2)$",
    )
    plt.xlabel("log(itetation)")
    plt.ylabel(r"$\log_{10}(||\theta-\eta_{*}||^2)$")
    plt.title(
        "Gradient descent vs Heavy Ball\n" + r"$\log_{10}(||\theta-\theta_{*}||^2$)"
    )
    plt.legend(loc="best", prop={"size": 9})
    plt.tight_layout()
    plt.savefig("images/HB_convex_log.pdf")
    plt.close()

    plt.plot(plot_range, np.log10(distances_to_opt_GD), label="GD")
    plt.plot(plot_range, np.log10(distances_to_opt_HB), label="Heavy Ball")
    plt.plot(
        plot_range,
        np.log10(upper_bounds_GD),
        label="upper bound, strongly convex loss function: "
        + r"$-\frac{2t}{\kappa}+\log_{10}(||\theta_0-\theta_{*}||^2)$",
    )
    plt.xlabel("iteration")
    plt.ylabel(r"$\log_{10}(||\theta-\theta_{*}||^2)$")
    plt.title(
        "Gradient descent vs Heavy Ball\n" + r"$\log_{10}(||\theta-\theta_{*}||^2$)"
    )
    plt.tight_layout()
    plt.legend(loc="best", prop={"size": 9})
    plt.savefig("images/HB_convex_semilog.pdf")
    plt.close()


if __name__ == "__main__":
    main()
