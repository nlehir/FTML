"""
    Gradient descent (GD) on a convex
    loss function.
    The design matrix is randomly generated.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from TP4_utils import OLS_estimator, gradient, error, square_distance_to_optimal_parameter, generate_output_data, upper_bound_convex, upper_bound_strongly_convex

"""
    Load the data
"""
n = 80
d = 60
X = np.load(f"./data/X_diagonal_n={n}_d={d}.npy")
rank = np.linalg.matrix_rank(X)
print(f"shape of X: {X.shape}")
print(f"rank of X: {rank}")
# generate output data
sigma = 0
r_state = 6
r = np.random.RandomState(r_state)
eta_star = r.uniform(-1, 1, size=(d, 1))
y = generate_output_data(X, eta_star, sigma, r)


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
beta = ((math.sqrt(L)-math.sqrt(mu))/(math.sqrt(L)+math.sqrt(mu)))**2
distances_to_opt_GD = list()
distances_to_opt_HB = list()
upper_bounds_GD = list()
upper_bounds_HB = list()
iteration_range = range(1, number_of_iterations)
gamma_HB = 10
gamma_GD = 10


"""
    Algorithm
"""
# perform one update
theta_GD = theta_0.copy()
theta_HB = theta_0.copy()
theta_HB_before = theta_0.copy()
theta_HB-=gamma_HB*gradient(theta_HB, H, X, y)
"""
    Add code here
"""


"""
    Plot the results
"""
