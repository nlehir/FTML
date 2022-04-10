"""
    Gradient descent (GD) on a strongly convex
    loss function.
    The design matrix is randomly generated.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from TP_3_utils import OLS_estimator, gradient, error, square_distance_to_optimal_parameter, upper_bound_strongly_convex, compute_gamma_star


"""
    Load the data
"""
n = 60
d = 40
X = np.load(f"./data/X_gaussian_n={n}_d={d}.npy")
y = np.load(f"./data/y_n={n}_d={d}.npy")


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
eigenvectors = eigenvectors[sorted_indexes]
# compute strong convexity and smoothness
L = eigenvalues[0]
mu = eigenvalues[-1]
kappa = L/mu
print(f"L: {L}")
print(f"mu: {mu}")
print(f"kappa: {kappa}")
# OLS estimator
eta_star = OLS_estimator(X, y)


"""
    Preparation of the algorithm
"""
theta_0 = np.zeros((d, 1))
number_of_iterations = 5000
gamma = 1/L
GD_distances_to_opt = list()
LS_distances_to_opt = list()
upper_bounds = list()
theta_GD = theta_0.copy()
theta_LS = theta_0.copy()
iteration_range = range(number_of_iterations)


"""
    Algorithm
    Run constant step-size GD
    and GD Line seach in parallel
"""
for iteration in iteration_range:
    GD_distances_to_opt.append(square_distance_to_optimal_parameter(theta_GD, eta_star))
    LS_distances_to_opt.append(square_distance_to_optimal_parameter(theta_LS, eta_star))
    upper_bounds.append(upper_bound_strongly_convex(iteration, kappa, theta_0, eta_star))
    # GD update
    theta_GD -= gamma*gradient(theta_GD, H, X, y)
    # GD Line search update
    grad = gradient(theta_LS, H, X, y)
    gamma_star = compute_gamma_star(H, grad)
    theta_LS -= gamma_star*gradient(theta_LS, H, X, y)


"""
    Plot the results
"""
# shift the plot range for visibility
# plot logarithm of the distance to optimal estimator
plot_range = [x+1 for x in iteration_range]
plt.plot(plot_range, np.log10(GD_distances_to_opt), label="GD")
plt.plot(plot_range, np.log10(LS_distances_to_opt), label="Line search")
plt.plot(plot_range, np.log10(upper_bounds), label="upper bound, strongly convex loss function: "+r"$-\frac{2t}{\kappa}+\log_{10}(||\theta_0-\eta_{*}||^2)$")
plt.xlabel("iteration")
plt.ylabel(r"$\log_{10}(||\theta-\eta_{*}||^2)$")
plt.ylabel(r"$||\theta-\eta_{*}||^2$")
plt.title("Constant step-size gradient descent vs exact line search\n"+r"$||\theta-\eta_{*}||^2$")
plt.legend(loc="best", prop={"size": 9})
plt.tight_layout()
plt.savefig("images_GD/LS_strongly_convex_semilog.pdf")
plt.close()

# plot the distance to optimal estimator
plt.plot(plot_range, GD_distances_to_opt, label="GD")
plt.plot(plot_range, LS_distances_to_opt, label="Line search")
plt.plot(plot_range, upper_bounds, label="upper bound, strongly convex loss function: "+r"$||\theta_0-\eta_{*}||^2\exp(\frac{-2t}{\kappa})$")
plt.xlabel("t")
plt.ylabel(r"$||\theta-\eta_{*}||^2$")
plt.title("Constant step-size gradient descent vs exact line search\n"+r"$||\theta-\eta_{*}||^2$")
plt.legend(loc="best", prop={"size": 9})
plt.tight_layout()
plt.savefig("images_GD/LS_strongly_convex.pdf")
