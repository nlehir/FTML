import numpy as np
import matplotlib.pyplot as plt
import os
import cProfile
import sys
from time import time

from TP5_LR_utils import empirical_risk, compute_accuracy

conv_tolerance = 0.99

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient_one_sample(theta, x, y, mu):
    d = len(theta)
    n = len(x)
    grad = -(x * y * sigmoid(-x.dot(theta) * y)).reshape(d, 1) + 1 / n * mu * theta
    return grad


def batch_gradient(theta, X, Y, mu):
    n, d = X.shape
    products = (X @ theta) * Y
    gradient = 1 / n * (X.T) @ (-Y * sigmoid(-products)) + mu * theta
    return gradient


def GD(
    theta_gd,
    gamma,
    X_train,
    y_train,
    X_test,
    y_test,
    n_iterations,
    scikit_empirical_risk,
    mu,
):
    empirical_risks_gd = list()
    test_errors_gd = list()
    risk_computations = list()
    # setup profiler
    profiler = cProfile.Profile()
    profiler.enable()
    tic = time()
    # setup iterations where we monitor empirical risk
    # we dont do it at each iteration to save time
    nb_empirical_risk_computations = 40
    step = int(n_iterations / nb_empirical_risk_computations)
    n_empirical_risk_computations = [step * k for k in range(int(n_iterations / step))]
    print("\nGD")
    # algorithm
    for iteration in range(n_iterations):
        # compute only some empirical risks in order to save time
        if iteration in n_empirical_risk_computations:
            print(f"iteration: {iteration}/{n_iterations}")
            empirical_risk_iter = empirical_risk(theta_gd, X_train, y_train)
            empirical_risks_gd.append(empirical_risk_iter)
            test_errors_gd.append(compute_accuracy(theta_gd, X_test, y_test))
            risk_computations.append(iteration)
            # early stopping
            if empirical_risk_iter < conv_tolerance * scikit_empirical_risk:
                print(f"attained {conv_tolerance:.2f}X scikit risk in {iteration} iterations")
                break
        theta_gd -= gamma * batch_gradient(theta_gd, X_train, y_train, mu)
    toc = time()
    gd_time = toc - tic
    print("exit GD")
    # log profile
    profiler.disable()
    stats_file = open(f"profiling/profiling_GD.txt", "w")
    sys.stdout = stats_file
    profiler.print_stats(sort="time")
    sys.stdout = sys.__stdout__
    # return results
    return theta_gd, empirical_risks_gd, test_errors_gd, gd_time, risk_computations


def SGD(
    theta_sgd,
    gamma,
    X_train,
    y_train,
    X_test,
    y_test,
    n_iterations,
    scikit_empirical_risk,
    mu,
):
    empirical_risks_sgd = list()
    test_errors_sgd = list()
    risk_computations = list()
    # setup iterations where we monitor empirical risk
    # we dont do it at each iteration to save time
    nb_empirical_risk_computations = 40
    step = int(n_iterations / nb_empirical_risk_computations)
    n_empirical_risk_computations = [step * k for k in range(int(n_iterations / step))]
    # setup profiler
    profiler = cProfile.Profile()
    profiler.enable()
    tic = time()
    # average = theta_sag.copy()
    print("\nSGD")
    n_train = X_train.shape[0]
    for iteration in range(n_iterations):
        if iteration in n_empirical_risk_computations:
            print(f"iteration: {iteration}/{n_iterations}")
            empirical_risk_iter = empirical_risk(theta_sgd, X_train, y_train)
            empirical_risks_sgd.append(empirical_risk_iter)
            test_errors_sgd.append(compute_accuracy(theta_sgd, X_test, y_test))
            risk_computations.append(iteration)
            # early stopping
            if empirical_risk_iter < conv_tolerance * scikit_empirical_risk:
                print(f"attained {conv_tolerance:.2f}X scikit risk in {iteration} iterations")
                break
        index = np.random.randint(n_train)
        x = X_train[index]
        y = y_train[index]
        theta_sgd -= gamma * gradient_one_sample(theta_sgd, x, y, mu)
    toc = time()
    sgd_time = toc - tic
    print("exit SGD")
    # log profile
    profiler.disable()
    stats_file = open(f"profiling/profiling_SGD.txt", "w")
    sys.stdout = stats_file
    profiler.print_stats(sort="time")
    # return results
    sys.stdout = sys.__stdout__
    return theta_sgd, empirical_risks_sgd, test_errors_sgd, sgd_time, risk_computations


def SAG(
    theta_sag,
    gamma,
    X_train,
    y_train,
    X_test,
    y_test,
    n_iterations,
    scikit_empirical_risk,
    mu,
):
    empirical_risks_sag = list()
    test_errors_sag = list()
    risk_computations = list()
    n_train, d_train = X_train.shape
    estimates = np.zeros((d_train, n_train))
    gradient_estimate = np.zeros((d_train, 1))
    # setup iterations where we monitor empirical risk
    # we dont do it at each iteration to save time
    nb_empirical_risk_computations = 40
    step = int(n_iterations / nb_empirical_risk_computations)
    n_empirical_risk_computations = [step * k for k in range(int(n_iterations / step))]
    # setup profiler
    profiler = cProfile.Profile()
    profiler.enable()
    tic = time()
    print("\nsag")
    n_train = X_train.shape[0]
    for iteration in range(n_iterations):
        if iteration in n_empirical_risk_computations:
            print(f"iteration: {iteration}/{n_iterations}")
            empirical_risk_iter = empirical_risk(theta_sag, X_train, y_train)
            empirical_risks_sag.append(empirical_risk_iter)
            test_errors_sag.append(compute_accuracy(theta_sag, X_test, y_test))
            risk_computations.append(iteration)
            # early stopping
            if empirical_risk_iter < conv_tolerance * scikit_empirical_risk:
                print(f"attained {conv_tolerance:.2f}X scikit risk in {iteration} iterations")
                break
        """
            EDIT HERE TO PERFORM THE ALGORITHM
        """
    toc = time()
    sag_time = toc - tic
    print("exit sag")
    # log profile
    profiler.disable()
    stats_file = open(f"profiling/profiling_sag.txt", "w")
    sys.stdout = stats_file
    profiler.print_stats(sort="time")
    # return results
    sys.stdout = sys.__stdout__
    return theta_sag, empirical_risks_sag, test_errors_sag, sag_time, risk_computations
