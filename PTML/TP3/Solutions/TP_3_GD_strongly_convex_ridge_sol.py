"""
    Gradient descent (GD) on a strongly convex
    loss function.
    The design matrix is randomly generated.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from TP_3_utils import OLS_estimator, gradient, loss_ridge, square_distance_to_optimal_parameter, upper_bound_strongly_convex, generate_output_data


"""
    Load the data
"""
n = 80
d = 100
X = np.load(f"./data/X_gaussian_n={n}_d={d}.npy")
sigma = 0
r_state = 6
r = np.random.RandomState(r_state)
eta_star = r.uniform(-1, 1, size=(d, 1))
y = generate_output_data(X, eta_star, sigma, r)

# regularization parameter
nu = 1

"""
    Compute the important quantities
"""
# Hessian matrix + regularization
G = 1/n*np.matmul(np.transpose(X), X)+nu*np.identity(d)
# compute spectrum of G
eigenvalues, eigenvectors = np.linalg.eig(G)
# sort the eigenvalues
sorted_indexes = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[sorted_indexes]
eigenvectors = eigenvectors[sorted_indexes]
# compute strong convexity and smoothness
L = eigenvalues[0]
mu = eigenvalues[-1]
kappa = L/mu
print(f"L: {L}")
print(f"mu: {mu}")
print(f"kappa: {kappa}")


"""
    Preparation of the algorithm
"""
theta_0 = np.zeros((d, 1))
number_of_iterations = 50
gamma = 1/L
losses = list()
theta = theta_0
iteration_range = range(1, number_of_iterations)


"""
    Algorithm
"""
for iteration in iteration_range:
    losses.append(loss_ridge(theta, X, y, nu))
    theta -= gamma*gradient(theta, G, X, y)


"""
    Plot the results
"""
plot_range = iteration_range
plt.plot(np.log10(plot_range), np.log10(losses), label="GD")
plt.xlabel(r"$\log(t)$")
plt.ylabel(r"$\log_{10}(f(\theta))$")
plt.title("Gradient descent on ridge regression\n"+r"$\log_{10}(f(\theta))$")
plt.legend(loc="best", prop={"size": 9})
plt.tight_layout()
plt.savefig("images_GD/GD_ridge.pdf")
plt.close()

plt.plot(plot_range, np.log10(losses), label="GD")
plt.xlabel(r"t")
plt.ylabel(r"$\log_{10}(f(\theta))$")
plt.title("Gradient descent on ridge regression\n" + r"$\log_{10}(f(\theta))$"+"\nsemilog")
plt.tight_layout()
plt.legend(loc="best", prop={"size": 9})
plt.savefig("images_GD/GD_ridge_semilog.pdf")
plt.close()
