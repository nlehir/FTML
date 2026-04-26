"""
Template of functions to edit to fulfill the exercises.
If you prefer, you can also start from scratch and write your own functions.
"""

import os

import matplotlib.pyplot as plt
from constants import BAYES_RISK, SIGMA


def plot_test_errors_ols(
    test_errors: dict[int, float],
    n_list: list[int],
    d_list: list[int],
    n_repetitions: int,
    statistical_setting: str,
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
        if statistical_setting == "fixed_design_gaussian":
            risks_theory = [BAYES_RISK + SIGMA**2 * d / n for n in n_list]
        elif statistical_setting == "random_design_gaussian":
            risks_theory = [
                BAYES_RISK + (SIGMA**2 * d / n) * (1 / (1 - (d + 1) / n))
                for n in n_list
            ]
        else:
            messsage = (
                f"Unknown statistical_setting {statistical_setting}! "
                "Should be one of 'random_design_gaussian', 'fixed_design_gaussian'"
            )
            raise ValueError(messsage)
        alpha = 0.6
        # extended label for the first ont
        if index == 0:
            label_est = f"test error, d={d}"
            if statistical_setting == "fixed_design_gaussian":
                label_th = r"$\sigma^2+\frac{\sigma^2d}{n}$" + f", d={d}"
            elif statistical_setting == "random_design_gaussian":
                label_th = (
                    r"$\sigma^2+\frac{\sigma^2d}{n}\frac{1}{1-(d+1)/n}$" + f", d={d}"
                )
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

    plt.xlabel("number of samples in the train set")
    plt.ylabel("test error")
    title = (
        f"OLS: test errors as a function of n and d\nn repetitions {n_repetitions}"
        f"\n{clean_filename(statistical_setting)}"
    )
    plt.title(title)
    plt.tight_layout()
    plt.legend(loc="best")

    # save plot
    fig_name = f"ols_test_errors_{statistical_setting}_{n_repetitions}_repetitions.pdf"
    # plt.yscale("log")
    plt.savefig(fig_name)
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


def clean_filename(name):
    name = name.replace("_", " ")
    return name
