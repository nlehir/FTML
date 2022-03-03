import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# if you want to clean the images
image_directory = "./images_ols"
for image in os.listdir(image_directory):
    os.remove(os.path.join(image_directory, image))

"""
    This scripts studies the OLS estimator:
        - risk
        - dependence of the risk on the dimensions n and d
        - stability of the OLS estimator
"""


def generate_output_data(X, theta_star, sigma):
    """
        generate input and output data (supervised learning)
        according to the linear model, fixed design setup
        - X is fixed
        - Y is random, according to

        Y = Xtheta_star + epsilon

        where epsilon is a centered gaussian noise vector with variance
        sigma*In

        Parameters:
            X (float matrix): (n, d) design matrix
            theta_star (float vector): (d, 1) vector (optimal parameter)
            sigma (float): variance each epsilon

        Returns:
            Y (float matrix): output vector (n, 1)
    """

    # output data
    n = X.shape[0]
    d = X.shape[1]
    noise = r.normal(0, sigma, size=(n, 1))
    Y = np.matmul(X, theta_star)+noise
    return Y


def OLS_estimator(X, Y):
    """
        Compute the OLS estimator from the data.

        Parameters:
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector

        Returns:
            OLS (float vector): (d, 1) vector
    """
    covariance_matrix = np.matmul(np.transpose(X), X)
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = np.matmul(inverse_covariance, np.matmul(np.transpose(X), Y))
    return theta_hat


def distance_to_theta_star(theta_hat, theta_star):
    """
        Relative distance between theta_hat and theta_star
    """
    return np.linalg.norm(theta_hat - theta_star)/np.linalg.norm(theta_star)


def error(theta_hat, X, Y):
    """
        Compute the prediction error with parameter theta_hat,
        between Xtheta_hat and the labels Y

        Parameters:
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector
            theta_hat (float vector): (d, 1) vector

        Returns:
            Mean square error
    """
    n_samples = X.shape[0]
    Y_predictions = np.matmul(X, theta_hat)
    return 1/n_samples*(np.linalg.norm(Y-Y_predictions))**2


def ols_risk(n, d, n_tests, plot=True):
    """
        Statistical evaluation of the excess risk of the OLS estimator.
        n_test times, do:
            - Draw output vector Y, according to the linear model, fixed
            design setup.
            - split the dataset into a train set and a test set
            - use the train set compute the OLS estimator
            - use the test test in order to have an estimation of the risk of
            this estimator (generalization error)

        In order to test the estimator and estimate the risk,
        we need to use a different output vector Y_test,
        that has not been used for training.

        Parameters:
            n (int): number of samples in the dataset
            d (int): dimension of each sample (number of features)
            n_tests (int): number of simulations run

        Returns:
            risk_estimation (float): estimation of the excess risk of the OLS
            estimator in this setup.

    """
    # design matrix
    X = r.rand(n, d)
    # Different design matrix
    # X = np.load("data/design_matrix.npy")
    # n, d = X.shape

    # Bayes predictor
    theta_star = r.rand(d).reshape(d, 1)

    # data structures to store results
    empirical_risks = list()
    test_errors = list()
    distances_to_theta_star = list()
    estimators = np.zeros((d, n_tests))

    # run several simulations to have an estimation of the excess risk
    for i in range(n_tests):
        Y = generate_output_data(X, theta_star, sigma)
        # compute the OLS estimator
        theta_hat = OLS_estimator(X, Y)
        # compute and store empirical risk (train error)
        empirical_risks.append(error(theta_hat, X, Y))
        # generate test data
        nb_test_samples = int(n/2)
        X_test = X[:nb_test_samples, :]
        Y_test = generate_output_data(X_test, theta_star, sigma)
        # compute and store test error
        test_errors.append(error(theta_hat, X_test, Y_test))
        # store the distance between the OLS estimator and theta_star
        distances_to_theta_star.append(
            distance_to_theta_star(theta_hat, theta_star))
        estimators[:, i] = theta_hat.reshape(d)
    # average test errors to have an etimation of the risk
    risk_estimation = sum(test_errors)/n_tests
    mean_estimator = np.mean(estimators, axis=1).reshape(d, 1)
    distance_mean_estimator = distance_to_theta_star(
        mean_estimator, theta_star)

    # Plot some results
    if plot == True:
        # plot risks
        plt.plot(range(n_tests), empirical_risks, "o",
                 label="empirical risk", markersize=3, alpha=0.5)
        plt.plot(range(n_tests), test_errors, "o",
                 label="test errors", markersize=3, alpha=0.5)
        plt.plot(range(n_tests), [bayes_risk]*n_tests,
                 label="Bayes risk", color="aqua")
        plt.plot(range(n_tests), [risk_estimation]*n_tests,
                 label="Mean test error (risk estimation)", color="darkcyan")
        plt.xlabel("test index")
        plt.xticks(range(1, n_tests, int(n_tests/10)))
        plt.legend(loc="best")
        plt.title(f"empirical risk, test error, Bayes risk\nn={n}, d={d}")
        plt.savefig(f"images_ols/risks_n={n}_d={d}.pdf")
        plt.close()

        # plot distance to theta_star
        plt.plot(range(n_tests), distances_to_theta_star, "o",
                 label=r"$\frac{||\hat{\theta}-\theta^*||}{||\theta^*||}$", markersize=3, alpha=0.6)
        plt.xlabel("test index")
        plt.xticks(range(1, n_tests, int(n_tests/10)))
        plt.legend(loc="best")
        title = (r"$\frac{||<\hat{\theta}>-\theta^*||}{||\theta^*||}=$"
                 f"{distance_mean_estimator:.3E}"
                 f"\nn={n}, d={d}"
                 )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"images_ols/variation_n={n}_d={d}.pdf")
        plt.close()

    return risk_estimation


# dimension of the problem
n_list = range(30, 400, 60)
n_list = [30]
d_list = [2, 5, 10, 20]
d_list = [10]

# amount of noise (linear model, fixed design)
sigma = 0.2
bayes_risk = sigma**2

# use a seed to have consistent resutls
r = np.random.RandomState(4)

# Different design matrix
# X = np.load("data/design_matrix.npy")
# n, d = X.shape

# number of tests to estimate the excess risk
n_tests = 200

# Assess the influence of different values of n and d
risks = dict()
for n in n_list:
    for d in d_list:
        risks[(n, d)] = ols_risk(n, d, n_tests, plot=True)

colors = ["blue", "green", "darkred", "mediumvioletred", "darkmagenta"]
index = 0
for d in d_list:
    color = colors[index]
    print(d)
    risk_estimates = [risks[n, d] for n in n_list]
    risks_theory = [bayes_risk+sigma**2*d/n for n in n_list]
    alpha = 0.6
    if index == 0:
        label_est = f"estimate, d={d}"
        label_th = "theory: "+r"$\sigma^2+\frac{\sigma^2d}{n}$"+f", d={d}"
        plt.plot(n_list, risk_estimates, "o", label=label_est, color=color, markersize=3, alpha=alpha)
        plt.plot(n_list, risks_theory, label=label_th, color=color, alpha=alpha)
    else:
        label_est = f"d={d}"
        plt.plot(n_list, risk_estimates, "o", label=label_est, color=color, markersize=3, alpha=alpha)
        plt.plot(n_list, risks_theory, color=color, alpha=alpha)
    index += 1
plt.xlabel("n")
plt.ylabel("risk")
plt.plot(n_list, [bayes_risk]*len(n_list),
         label="Bayes risk: "+r"$\sigma^2$", color="aqua")
plt.title("OLS: risks as a function of n and d")
plt.legend(loc="best")
# plt.savefig("images_ols/risks.pdf")
