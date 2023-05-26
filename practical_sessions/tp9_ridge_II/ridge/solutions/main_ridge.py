"""
    Statistical comparison between Ridge regression estimator and OLS.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from utils_algo import compute_lambda_star_and_risk_star, ridge_risk
from constants import BAYES_RISK, SIGMA, SEED, N_TESTS, rng

def main():
    n = 30
    d_list = [10, 20, 30]
    llambda_list = [10**(n) for n in np.arange(-8, 5, 0.2)]
    # number of tests to estimate the excess risk

    risks = dict()
    llambda_stars_risks = dict()
    infinity_biases = dict()
    for llambda in llambda_list:
        for d in d_list:
            # Load design matrix
            X_path = os.path.join("data", f"n_design_matrix_n={n}_d={d}.npy")
            X = np.load(X_path)
            n = X.shape[0]

            # lecun initialisation of theta_star
            # theta_star_range = 1/math.sqrt(d)
            # theta_star = r.uniform(-theta_star_range, theta_star_range, size=(d, 1))

            # initialisation of theta_star with eigenvalues
            Sigma_matrix = 1/n*X.T @ X
            eigenvalues, eigenvectors = np.linalg.eig(Sigma_matrix)
            largest_eigenvalue_index = np.argmax(eigenvalues)
            largest_eigenvector = eigenvectors[:, largest_eigenvalue_index]
            theta_star = largest_eigenvector.reshape(d, 1)
            # theta_star = rng.uniform(size=(d, 1))

            # compute risk of the ridge estimator
            print(f"\nlambda: {llambda}")
            print(f"d: {d}")
            risks[(llambda, d)] = ridge_risk(
                    lambda_ = llambda,
                    n_tests = N_TESTS,
                    theta_star = theta_star,
                    X= X,
                    )

            # compute lambda_star and the corresponding risk
            llambda_stars_risks[d] = compute_lambda_star_and_risk_star(SIGMA, X, theta_star)

            # compute bias limit when llambda is large
            infinity_bias = (theta_star.T @ Sigma_matrix) @ theta_star
            infinity_biases[d] = infinity_bias[0]


    colors = ["blue", "green", "darkred", "mediumvioletred", "darkmagenta"]
    index = 0
    for d in d_list:
        color = colors[index]
        risk_estimates = [risks[llambda, d] for llambda in llambda_list]
        llambda_star, risk_star = llambda_stars_risks[d]

        # plot lambda_star and the corresponding risk
        plt.plot(llambda_star, risk_star, "x", color=color, markersize=12, label = r"$\lambda^*$"+f", d={d}")
        infinity_bias = infinity_biases[d]
        alpha = 0.4
        if index == 0:
            label_est = f"risk estimation, d={d}"
            plt.plot(llambda_list,
                     risk_estimates,
                     "o",
                     label=label_est,
                     color=color,
                     markersize=3,
                     alpha=alpha)
            plt.plot(llambda_list,
                     [BAYES_RISK+SIGMA**2*d/n]*len(llambda_list),
                     label="OLS risk: "+r"$\sigma^2+\frac{\sigma^2d}{n}$"+f", d={d}",
                     color=color,
                     alpha = alpha)
            plt.plot(llambda_list,
                     [SIGMA**2+infinity_bias]*len(llambda_list),
                     label=r"$Risk_{\lambda\rightarrow +\infty}$"+f", d={d}",
                     color=color,
                     alpha = 0.8*alpha,
                     linestyle="dashed")
        else:
            label_est = f"d={d}"
            plt.plot(llambda_list, risk_estimates, "o", label=label_est, color=color, markersize=3, alpha=alpha)
            plt.plot(llambda_list, [BAYES_RISK+SIGMA**2*d/n]*len(llambda_list), color=color, alpha=alpha)
            plt.plot(llambda_list, [SIGMA**2+infinity_bias]*len(llambda_list),
                     label=r"$Risk_{\lambda\rightarrow +\infty}$"+f", d={d}",
                     color=color, alpha = 0.8*alpha, linestyle="dashed")
        index += 1

    # finalize plot
    plt.xlabel(r"$\lambda$")
    plt.xscale("log")
    plt.ylabel("risk")
    plt.plot(llambda_list, [BAYES_RISK]*len(llambda_list), label="Bayes risk: "+r"$\sigma^2$", color="aqua")
    title = (
        "Ridge regression: test error as a function of "
        r"$\lambda$"
        f" and d\nn={n}")
    plt.title(title)
    plt.legend(loc="best", prop={"size": 6})
    plt.tight_layout()
    figname = f"test_errors_n={n}_r_state_{SEED}.pdf"
    figpath = os.path.join("images", figname)
    plt.savefig(figpath)


if __name__ == "__main__":
    main()
