import math

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt


def generate_output_data(X, theta_star, sigma):
    """
        same as previously
    """

    # output data
    n = X.shape[0]
    d = X.shape[1]
    noise = np.random.normal(0, sigma, size=(n, 1))
    Y = np.matmul(X, theta_star)+noise
    return Y


def compute_lambda_star_and_risk_star(sigma, X, theta_star, n):
    """
        same as previously
    """
    Sigma = 1/n*np.matmul(np.transpose(X), X)
    trace = np.trace(Sigma)
    llambda_star = sigma*math.sqrt(trace)/(np.linalg.norm(theta_star)*math.sqrt(n))
    excess_risk_star  = sigma**2+(sigma*math.sqrt(trace)*np.linalg.norm(theta_star))/math.sqrt(n)
    return llambda_star, excess_risk_star


def error(classifier, X, Y):
    """
        Compute the prediction error with the classifier obtained by CV,

        Parameters:
            classifier (object):
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector

        Returns:
            Mean square error
    """
    n_samples = X.shape[0]
    Y_predictions = classifier.predict(X)
    return 1/n_samples*(np.linalg.norm(Y-Y_predictions))**2


# n = 30
n = 200
# d_list = [10, 15, 20, 25, 30]
d_list = [n*10 for n in range(10, 21)]

# amount of noise (linear model, fixed design)
sigma = 0.1

llambdas_cv = list()
test_errors = list()
llambdas_star = list()
excess_risk_star = list()
for d in d_list:
    X = np.load(f"data/design_matrix_n={n}_d={d}.npy")

    # lecun initialisation of theta_star
    # theta_star_range = 1/math.sqrt(d)
    # theta_star = np.random.uniform(-theta_star_range, theta_star_range, size=(d, 1))

    # initialisation of theta_star with eigenvalues
    Sigma = 1/n*np.matmul(np.transpose(X), X)
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    largest_eigenvalue_index = np.argmax(eigenvalues)
    largest_eigenvalue = eigenvalues[largest_eigenvalue_index]
    largest_eigenvector = eigenvectors[:, largest_eigenvalue_index]
    theta_star = largest_eigenvector.reshape(d, 1)

    y = generate_output_data(X, theta_star, sigma)

    llambda_list = [10**(n) for n in np.arange(-8, 5, 0.2)]
    classifier = RidgeCV(alphas = llambda_list).fit(X, y)

    llambdas_cv.append(classifier.alpha_)
    # generate test data
    Y_test = generate_output_data(X, theta_star, sigma)
    test_errors.append(error(classifier, X, Y_test))

    llambdas_star.append(compute_lambda_star_and_risk_star(sigma, X, theta_star, n)[0])
    excess_risk_star.append(compute_lambda_star_and_risk_star(sigma, X, theta_star, n)[0])


plt.plot(d_list, llambdas_cv, "o", label=r"$\lambda$"+" found by cross validation")
plt.plot(d_list, llambdas_star, "x", label=r"$\lambda^*$")
plt.xlabel("d")
plt.yscale("log")
plt.title(f"Ridge regression optimal hyperparameter\nn={n}")
plt.legend(loc="best")
plt.savefig(f"images_ridge/cross_validation/RidgeCV_lambda_n={n}.pdf")
plt.close()

plt.plot(d_list, test_errors, "o", label=f"test error, cross validation")
plt.plot(d_list, excess_risk_star, "x", label="Excess risk bound for "+r"$\lambda^*$")
plt.xlabel("d")
plt.yscale("log")
plt.title(f"Ridge regression test error for optimal hyperparameter\nn={n}")
plt.legend(loc="best")
plt.savefig(f"images_ridge/cross_validation/RidgeCV_score_n={n}.pdf")
