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

"""
    Plot the results
"""
