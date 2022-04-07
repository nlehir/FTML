"""
    Gradient descent (GD) on a strongly convex
    loss function.
    The design matrix is randomly generated.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from TP_3_utils import OLS_estimator, gradient, error, square_distance_to_optimal_parameter, upper_bound_strongly_convex


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
number_of_iterations = 10000
GD_distances_to_opt = list()
upper_bounds = list()
theta = theta_0.copy()
iteration_range = range(number_of_iterations)
gamma = 10


"""
    Algorithm
"""
for iteration in iteration_range:
    GD_distances_to_opt.append(square_distance_to_optimal_parameter(theta, eta_star))
    upper_bounds.append(upper_bound_strongly_convex(iteration, kappa, theta_0, eta_star))
    theta -= gamma*gradient(theta, H, X, y)


"""
    Plot the results
"""
