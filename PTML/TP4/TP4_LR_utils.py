import numpy as np
import matplotlib.pyplot as plt
import os
import cProfile
import sys
from time import time


"""
    Dataset preparation
"""

def add_bias(data):
    n = data.shape[0]
    return np.hstack((data, np.ones((n, 1))))


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def sign(x):
    if x > 0:
        return 1
    else:
        return 0


def empirical_risk(theta, X, y):
    n_samples = X.shape[0]
    estimations = (X @ theta).reshape(n_samples, 1)
    # print(f"theta norm: {np.linalg.norm(theta):.2f}")
    # print(f"estimations norm: {np.linalg.norm(estimations):.2f}")
    # print(f"max estimations: {np.max(np.abs(estimations)):.2f}")
    losses = np.log(1 + np.exp(-estimations*y))
    emp_risk = 1/n_samples*losses.sum()
    return emp_risk



def compute_accuracy(theta, X, Y):
    n_samples = X.shape[0]
    correct_predictions = 0
    # estimations = (X @ theta).reshape(n_samples, 1)
    for i in range(n_samples):
        x = X[i, :]
        y = Y[i, :]
        estimation = x.dot(theta)
        correct_predictions += sign(estimation*y)
    return correct_predictions/n_samples


"""
    Visualisation
"""

def visualise_predictions(theta, X, color, label):
    n_samples = X.shape[0]
    estimations = list()
    for i in range(n_samples):
        x = X[i, :]
        estimations.append(x.dot(theta))
    plt.plot(range(n_samples), estimations, "o", color=color, label=label)
