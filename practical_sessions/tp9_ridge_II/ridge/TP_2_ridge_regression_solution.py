"""
    Statistical comparison between Ridge regression estimator and OLS.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from utils_algo import ridge_regression_estimator, error

# if you want to clean the images
image_directory = "./images"
# for image in os.listdir(image_directory):
#     os.remove(os.path.join(image_directory, image))




def ridge_risk(theta_star, llambda, n, d, n_tests, X, plot=True):
    """
        Statistical evaluation of the excess risk of the Ridge estimator.

        n_test times, do:
            - Draw output vector Y, according to the linear model, fixed
            design setup.
            - compute the corresponding Ridge estimator
            - generate a test test in order to have an estimation of the excess risk of
            this estimator (generalization error)

        Parameters:
            n (int): number of samples in the dataset
            d (int): dimension of each sample (number of features)
            n_tests (int): number of simulations run
            llambda: scalar, regularization parameter

        Returns:
            risk_estimation (float): estimation of the excess risk of the Ridge
            estimator in this setup.

    """
    # data structures to store results
    empirical_risks = list()
    test_errors = list()

     # run several simulations to have an estimation of the excess risk
    for i in range(n_tests):
       Y = generate_output_data(X, theta_star, sigma)
       # compute the Ridge estimator
       theta_hat_llambda = ridge_regression_estimator(X, Y, llambda)
       # generate test data
       nb_test_samples = int(n/2)
       X_test = X[:nb_test_samples, :]
       Y_test = generate_output_data(X_test, theta_star, sigma)
       # compute and store test error
       test_errors.append(error(theta_hat_llambda, X_test, Y_test))
    # average test errors to have an etimation of the risk
    risk_estimation = sum(test_errors)/n_tests
    return risk_estimation


# number of samples
n = 30
# number of features
d_list = [10, 20, 30]

# amount of noise (linear model, fixed design)
sigma = 2
# bayes risk
bayes_risk = sigma**2

# regularization hyperparameters
llambda_list = [10**(n) for n in np.arange(-8, 5, 0.2)]

# use a seed to have consistent resutls
r_state = 10
r = np.random.RandomState(r_state)

# number of tests to estimate the excess risk
n_tests = 500

# Assess the influence of lambda in several dimensions
risks = dict()
llambda_stars_risks = dict()
infinity_biases = dict()
for llambda in llambda_list:
    for d in d_list:
        # Load design matrix
        X = np.load(f"data/design_matrix_n={n}_d={d}.npy")
        n = X.shape[0]

        # lecun initialisation of theta_star
        # theta_star_range = 1/math.sqrt(d)
        # theta_star = r.uniform(-theta_star_range, theta_star_range, size=(d, 1))

        # initialisation of theta_star with eigenvalues
        Sigma = 1/n*np.matmul(np.transpose(X), X)
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        largest_eigenvalue_index = np.argmax(eigenvalues)
        largest_eigenvalue = eigenvalues[largest_eigenvalue_index]
        largest_eigenvector = eigenvectors[:, largest_eigenvalue_index]
        theta_star = largest_eigenvector.reshape(d, 1)

        # compute risk of the ridge estimator
        risks[(llambda, d)] = ridge_risk(theta_star, llambda, n, d, n_tests, X, plot=True)

        # compute lambda_star and the corresponding risk
        llambda_stars_risks[d] = compute_lambda_star_and_risk_star(sigma, X, theta_star, n)

        # compute bias limit when llambda is large
        Sigma = 1/n*np.matmul(np.transpose(X), X)
        infinity_bias = np.matmul(np.matmul(np.transpose(theta_star), Sigma), theta_star)
        infinity_biases[d] = infinity_bias[0]


colors = ["blue", "green", "darkred", "mediumvioletred", "darkmagenta"]
index = 0
for d in d_list:
    color = colors[index]
    risk_estimates = [risks[llambda, d] for llambda in llambda_list]
    # plot lambda_star and the corresponding risk
    llambda_star, risk_star = llambda_stars_risks[d]
    plt.plot(llambda_star, risk_star, "x", color=color, markersize=12, label = r"$\lambda^*$"+f", d={d}")
    infinity_bias = infinity_biases[d]
    alpha = 0.4
    if index == 0:
        label_est = f"risk estimation, d={d}"
        plt.plot(llambda_list, risk_estimates, "o", label=label_est, color=color, markersize=3, alpha=alpha)
        plt.plot(llambda_list, [bayes_risk+sigma**2*d/n]*len(llambda_list), label="OLS risk: "+r"$\sigma^2+\frac{\sigma^2d}{n}$"+f", d={d}", color=color, alpha = alpha)
        plt.plot(llambda_list, [sigma**2+infinity_bias]*len(llambda_list), label=r"$Risk_{\lambda\rightarrow +\infty}$"+f", d={d}", color=color, alpha = 0.8*alpha, linestyle="dashed")
    else:
        label_est = f"d={d}"
        plt.plot(llambda_list, risk_estimates, "o", label=label_est, color=color, markersize=3, alpha=alpha)
        plt.plot(llambda_list, [bayes_risk+sigma**2*d/n]*len(llambda_list), color=color, alpha=alpha)
        plt.plot(llambda_list, [sigma**2+infinity_bias]*len(llambda_list),
                 label=r"$Risk_{\lambda\rightarrow +\infty}$"+f", d={d}",
                 color=color, alpha = 0.8*alpha, linestyle="dashed")
    index += 1
plt.xlabel(r"$\lambda$")
plt.xscale("log")
plt.ylabel("risk")
plt.plot(llambda_list, [bayes_risk]*len(llambda_list), label="Bayes risk: "+r"$\sigma^2$", color="aqua")
plt.title("Ridge regression: risks as a function of " + r"$\lambda$" + f" and d\nn={n}")
plt.legend(loc="best", prop={"size": 6})
plt.tight_layout()
plt.savefig(f"{image_directory}/set_1_risks_n={n}_r_state_{r_state}.pdf")
