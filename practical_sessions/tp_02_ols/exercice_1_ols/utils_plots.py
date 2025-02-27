"""
Template of functions to edit to fulfill the exercises.
If you prefer, you can also start from scratch and write your own functions.
"""

import os

import matplotlib.pyplot as plt

from constants import BAYES_RISK, SIGMA


def plot_test_errors_ols(
    test_errors: dict[tuple, float], n_list: list[int], d_list: list[int]
):
    """
    Display all the computed test_errors on a plot
    """
    colors = ["blue", "green", "darkred", "mediumvioletred", "darkmagenta"]
    index = 0

    # plot the test_errors for each n and d
    for index, d in enumerate(d_list):
        print(f"d={d}")
        color = colors[index]
        risk_estimates = [test_errors[n, d] for n in n_list]
        risks_theory = [BAYES_RISK + SIGMA**2 * d / n for n in n_list]
        alpha = 0.6
        # extended label for the first ont
        if index == 0:
            label_est = f"test error, d={d}"
            label_th = r"$\sigma^2+\frac{\sigma^2d}{n}$" + f", d={d}"
            plt.plot(
                n_list,
                risk_estimates,
                "o",
                label=label_est,
                color=color,
                markersize=3,
                alpha=alpha,
            )
            plt.plot(n_list, risks_theory, label=label_th, color=color, alpha=alpha)
        else:
            label_est = f"d={d}"
            plt.plot(
                n_list,
                risk_estimates,
                "o",
                label=label_est,
                color=color,
                markersize=3,
                alpha=alpha,
            )
            plt.plot(n_list, risks_theory, color=color, alpha=alpha)

    # plot bayes risk
    plt.plot(
        n_list,
        [BAYES_RISK] * len(n_list),
        label="Bayes risk: " + r"$\sigma^2$",
        color="aqua",
    )

    # finish plot
    plt.xlabel("n")
    plt.ylabel("test error")
    plt.title("OLS: test errors as a function of n and d")
    plt.legend(loc="best")

    # save plot
    fig_path = os.path.join("ols_test_errors.pdf")
    # plt.yscale("log")
    plt.savefig(fig_path)
    plt.close()


def plot_stds(stds: dict[tuple, float], n_list: list[int], d_list: list[int]):
    """
    Display the standard deviation of the relative distance of
    the OLS estimator to the Bayes estimator.
    """
    colors = ["blue", "green", "darkred", "mediumvioletred", "darkmagenta"]
    # plot the stds for each n and d
    for index, d in enumerate(d_list):
        color = colors[index]
        std = [stds[n, d] for n in n_list]
        alpha = 0.6
        label = f"d={d}"
        plt.plot(n_list, std, "o", label=label, color=color, markersize=3, alpha=alpha)

    # finish plot
    plt.xlabel("n")
    plt.ylabel("standard deviation")
    title = (
        "Standard deviation of\n" r"$\frac{||\hat{\theta}-\theta^*||}{||\theta^*||}$"
    )
    plt.title(title)
    plt.legend(loc="best")

    # save plot
    fig_path = os.path.join("images", "ols_stds.pdf")
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(fig_path)
    plt.close()
