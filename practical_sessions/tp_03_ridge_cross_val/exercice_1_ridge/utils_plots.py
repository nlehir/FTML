"""
Template of functions to edit to fulfill the exercises.
If you prefer, you can also start from scratch and write your own functions.
"""

import os

import matplotlib.pyplot as plt
from constants import BAYES_RISK, SIGMA

FONTSIZE = 10


def plot_test_errors_ols(
    risks: dict[tuple, float], n_list: list[int], d_list: list[int]
):
    """
    Display all the computed risks on a plot
    """
    colors = ["blue", "green", "darkred", "mediumvioletred", "darkmagenta"]
    index = 0

    # plot the risks for each n and d
    for index, d in enumerate(d_list):
        print(f"d={d}")
        color = colors[index]
        risk_estimates = [risks[n, d] for n in n_list]
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
    plt.ylabel("risk")
    plt.title("OLS: risks as a function of n and d")
    plt.legend(loc="best")

    # save plot
    fig_path = os.path.join("images", "ols_risks.pdf")
    plt.yscale("log")
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


def plot_test_errors_ridge(
    risks: dict[tuple, float],
    d_list: list[int],
    n: int,
    lambda_list: list[int],
    n_repetitions: int,
    theta_star_type: str,
    design_matrix_type: str,
) -> None:
    """
    Display all the computed risks on a plot
    """
    colors = ["blue", "green", "darkred", "mediumvioletred", "darkmagenta"]
    index = 0

    # plot the risks for each n and d
    for index, d in enumerate(d_list):
        print(f"d: {d}")
        color = colors[index]
        risk_estimates = [risks[d, lambda_] for lambda_ in lambda_list]
        ols_risk = BAYES_RISK + SIGMA**2 * d / n
        alpha = 0.6
        # extended label for the first ont
        label_est = f"Ridge test error, d={d}"
        label_ols = f"OLS risk, d={d}"
        plt.plot(
            lambda_list,
            risk_estimates,
            "o",
            label=label_est,
            color=color,
            markersize=3,
            alpha=alpha,
        )
        plt.plot(
            lambda_list,
            [ols_risk] * len(lambda_list),
            label=label_ols,
            color=color,
            alpha=alpha,
        )

    plt.plot(
        lambda_list,
        [BAYES_RISK] * len(lambda_list),
        label="Bayes risk: " + r"$\sigma^2$",
        color="aqua",
    )

    # finish plot
    plt.xlabel(r"$\lambda$", fontsize=FONTSIZE)
    plt.ylabel("test error", fontsize=FONTSIZE)
    title = (
        "Ridge regression: risks as a function of " + r"$\lambda$"
        f"\n{n} training samples"
        f"\nn repetitions {n_repetitions}"
        f"\n{design_matrix_type} design matrix"
        f"\n{theta_star_type} Bayes estimator"
    )
    plt.title(title, fontsize=FONTSIZE)
    plt.legend(loc="best", fontsize=6)

    # save plot
    fig_name = (
        f"ridge_risks_{n_repetitions}_repetitions"
        f"_{design_matrix_type}_X"
        f"_{theta_star_type}_theta"
    )
    fig_path = os.path.join(f"{fig_name}.pdf")

    plt.yscale("log")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
