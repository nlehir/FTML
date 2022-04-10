"""
    Gradient descent (GD) on a convex
    loss function.
    The design matrix is randomly generated.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from TP_3_utils import OLS_estimator, gradient, error, square_distance_to_optimal_parameter, generate_output_data, upper_bound_convex

"""
    Load the data
"""
n = 80
d = 60
X = np.load(f"./data/X_singular_n={n}_d={d}.npy")
rank = np.linalg.matrix_rank(X)
print(f"shape of X: {X.shape}")
print(f"rank of X: {rank}")
# generate output data
sigma = 0
r_state = 6
r = np.random.RandomState(r_state)
eta_star = r.uniform(-1, 1, size=(d, 1))
y = generate_output_data(X, eta_star, sigma, r)

# n = 40
# d = 20
# X = np.load(f"./data/X_n={n}_d={d}.npy")
# y = np.load(f"./data/y_n={n}_d={d}.npy")
# eta_star = OLS_estimator(X, y)


"""
    Compute the important quantities
"""
# Hessian matrix
H = 1/n*np.matmul(np.transpose(X), X)
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
kappa = L/mu
print(f"L: {L}")
print(f"mu: {mu}")
print(f"kappa: {kappa}")

"""
    Preparation of the algorithm
"""
theta_0 = np.zeros((d, 1))
number_of_iterations = 10000
gamma = 1/L
losses = list()
upper_bounds = list()
theta = theta_0.copy()
iteration_range = range(1, number_of_iterations)


"""
    Algorithm
"""
for iteration in iteration_range:
    losses.append(error(theta, X, y))
    upper_bounds.append(upper_bound_convex(iteration, kappa, theta_0, eta_star, gamma))
    theta -= gamma*gradient(theta, H, X, y)


"""
    Compute excess loss
"""
loss_star = error(eta_star, X, y)
excess_losses = [loss-loss_star for loss in losses]


"""
    Plot the results
"""
plot_range = iteration_range
plt.plot(np.log10(plot_range), np.log10(excess_losses), label="GD")
plt.plot(np.log10(plot_range), np.log10(upper_bounds), label="upper bound, convex loss function: " + r"$-\log(4\gamma t)+\log_{10}(||\theta_0-\eta_{*}||^2)$")
plt.xlabel(r"$\log(t)$")
plt.ylabel(r"$\log_{10}(f(\theta)-f(\eta_{*}))$")
plt.title("Gradient descent\n"+r"$\log_{10}(f(\theta)-f(\eta_{*}))$")
plt.legend(loc="best", prop={"size": 9})
plt.tight_layout()
plt.savefig("images_GD/GD_convex_log.pdf")
plt.close()

plt.plot(plot_range, np.log10(excess_losses), label="GD")
plt.plot(plot_range, np.log10(upper_bounds), label="upper bound, convex loss function: " +
         r"$-\log(4\gamma t)+\log_{10}(||\theta_0-\eta_{*}||^2)$")
plt.xlabel(r"t")
plt.ylabel(r"$\log_{10}(f(\theta)-f(\eta_{*}))$")
plt.title("Gradient descent\n" + r"$\log_{10}(f(\theta)-f(\eta_{*}))$"+"\nsemilog")
plt.tight_layout()
plt.legend(loc="best", prop={"size": 9})
plt.savefig("images_GD/GD_convex_semilog.pdf")
plt.close()
