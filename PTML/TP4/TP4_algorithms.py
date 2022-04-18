import numpy as np
import math
import matplotlib.pyplot as plt
import os
import cProfile
import sys
from time import time

from TP4_LR_utils import empirical_risk, compute_accuracy, sigmoid

conv_tolerance = 1.05


def gradient_one_sample(theta, x, y, mu):
    d = len(theta)
    n = len(x)
    grad = - (x * y * sigmoid(-x.dot(theta) * y)).reshape(d, 1)+mu*theta
    return grad


def batch_gradient(theta, X, Y, mu):
    n, d = X.shape
    products = (X@theta)*Y
    gradient = 1/n*(X.T)@(-Y*sigmoid(-products))+mu*theta
    return gradient


def learning_rate_schedule(gamma_0, iteration, schedule):
    """
        Define the learning rate schedule.
    """
    if schedule == "decreasing 1":
        """
            Theoretical rate for convex losses
        """
        iteration_scale = 100
        return gamma_0/(1+iteration/iteration_scale)
    if schedule == "decreasing 2":
        """
            Theoretical rate for strongly convex losses
            It is slightly larger than the previous one
        """
        iteration_scale = 100
        return gamma_0/(1+math.sqrt(iteration/iteration_scale))
    if schedule == "constant":
        """
            Constant learning rate
        """
        return gamma_0
    else:
        raise ValueError("unknown learning rate schedule")


def GD(theta_gd, gamma, X_train, y_train, X_test, y_test, n_iterations, scikit_empirical_risk, mu):
    nb_empirical_risk_computations = n_iterations
    nb_empirical_risk_computations = 10
    empirical_risks_gd = list()
    test_errors_gd = list()
    risk_computations = list()
    # setup profiler
    profiler = cProfile.Profile()
    profiler.enable()
    tic = time()
    # setup iterations where we monitor empirical risk
    # we dont do it at each iteration to save time
    step = int(n_iterations/nb_empirical_risk_computations)
    n_empirical_risk_computations = [
        step*k for k in range(int(n_iterations/step))]
    print("\nGD")
    # algorithm
    for iteration in range(n_iterations):
        # compute only some empirical risks in order to save time
        if iteration in n_empirical_risk_computations:
            empirical_risk_iter = empirical_risk(theta_gd, X_train, y_train)
            empirical_risks_gd.append(empirical_risk_iter)
            test_errors_gd.append(compute_accuracy(theta_gd, X_test, y_test))
            risk_computations.append(iteration)
            print(f"iteration: {iteration}/{n_iterations}")
            print(f"loss: {empirical_risk_iter:.2f}")
            # early stopping
            if empirical_risk_iter < conv_tolerance*scikit_empirical_risk:
                print(f"attained {conv_tolerance} X scikit risk in {iteration} iterations")
                print("exit GD")
                break
        theta_gd -= gamma*batch_gradient(theta_gd, X_train, y_train, mu)
    toc = time()
    gd_time = toc-tic
    # log profile
    profiler.disable()
    stats_file = open(f"profiling_GD.txt", 'w')
    sys.stdout = stats_file
    profiler.print_stats(sort='time')
    sys.stdout = sys.__stdout__
    # return results
    return theta_gd, empirical_risks_gd, test_errors_gd, gd_time, risk_computations


def SGD(theta_sgd, gamma_0, X_train, y_train, X_test, y_test, n_iterations, scikit_empirical_risk, mu, schedule):
    nb_empirical_risk_computations = n_iterations
    nb_empirical_risk_computations = 10
    empirical_risks_sgd = list()
    test_errors_sgd = list()
    risk_computations = list()
    # setup iterations where we monitor empirical risk
    # we dont do it at each iteration to save time
    step = int(n_iterations/nb_empirical_risk_computations)
    n_empirical_risk_computations = [
        step*k for k in range(int(n_iterations/step))]
    # setup profiler
    profiler = cProfile.Profile()
    profiler.enable()
    tic = time()
    # average = theta_sag.copy()
    print("\nSGD")
    n_train = X_train.shape[0]
    for iteration in range(n_iterations):
        if iteration in n_empirical_risk_computations:
            empirical_risk_iter = empirical_risk(theta_sgd, X_train, y_train)
            empirical_risks_sgd.append(empirical_risk_iter)
            test_errors_sgd.append(compute_accuracy(theta_sgd, X_test, y_test))
            risk_computations.append(iteration)
            print(f"iteration: {iteration}/{n_iterations}")
            print(f"loss: {empirical_risk_iter:.2f}")
            # print(f"theta norm: {np.linalg.norm(theta_sgd):.2f}")
            # early stopping
            if empirical_risk_iter < conv_tolerance*scikit_empirical_risk:
                print(f"attained {conv_tolerance} X scikit risk in {iteration} iterations")
                print("exit SGD")
                break
        gamma = learning_rate_schedule(gamma_0, iteration, schedule)
        index = np.random.randint(n_train)
        x = X_train[index]
        y = y_train[index]
        theta_sgd -= gamma*gradient_one_sample(theta_sgd, x, y, mu)
    toc = time()
    sgd_time = toc-tic
    # log profile
    profiler.disable()
    stats_file = open(f"profiling_SGD.txt", 'w')
    sys.stdout = stats_file
    profiler.print_stats(sort='time')
    # return results
    sys.stdout = sys.__stdout__
    return theta_sgd, empirical_risks_sgd, test_errors_sgd, sgd_time, risk_computations


def SGA(theta_sga, gamma_0, X_train, y_train, X_test, y_test, n_iterations, scikit_empirical_risk, mu, schedule):
    nb_empirical_risk_computations = n_iterations
    nb_empirical_risk_computations = 10
    empirical_risks_sga = list()
    test_errors_sga = list()
    risk_computations = list()
    # setup iterations where we monitor empirical risk
    # we dont do it at each iteration to save time
    step = int(n_iterations/nb_empirical_risk_computations)
    n_empirical_risk_computations = [
        step*k for k in range(int(n_iterations/step))]
    # setup profiler
    profiler = cProfile.Profile()
    profiler.enable()
    tic = time()
    # average = theta_sag.copy()
    print("\nsga")
    n_train = X_train.shape[0]
    theta_sgd = theta_sga.copy()
    for iteration in range(n_iterations):
        if iteration in n_empirical_risk_computations:
            empirical_risk_iter = empirical_risk(theta_sga, X_train, y_train)
            empirical_risks_sga.append(empirical_risk_iter)
            test_errors_sga.append(compute_accuracy(theta_sga, X_test, y_test))
            risk_computations.append(iteration)
            print(f"iteration: {iteration}/{n_iterations}")
            print(f"loss: {empirical_risk_iter:.2f}")
            # print(f"theta norm: {np.linalg.norm(theta_sga):.2f}")
            # early stopping
            if empirical_risk_iter < conv_tolerance*scikit_empirical_risk:
                print(f"attained {conv_tolerance} X scikit risk in {iteration} iterations")
                print("exit sga")
                break
        gamma = learning_rate_schedule(gamma_0, iteration, schedule)
        index = np.random.randint(n_train)
        x = X_train[index]
        y = y_train[index]
        theta_sgd -= gamma*gradient_one_sample(theta_sgd, x, y, mu)
        # averaging
        theta_sga = 1/(iteration+1)*theta_sgd+iteration/(iteration+1)*theta_sga
    toc = time()
    sga_time = toc-tic
    # log profile
    profiler.disable()
    stats_file = open(f"profiling_sga.txt", 'w')
    sys.stdout = stats_file
    profiler.print_stats(sort='time')
    # return results
    sys.stdout = sys.__stdout__
    return theta_sga, empirical_risks_sga, test_errors_sga, sga_time, risk_computations
