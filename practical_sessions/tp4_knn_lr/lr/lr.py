"""
Use first-order methods (GD, SGD) and scikit-learn in order to perform
logistic regression on a simple example.
"""

import os
from time import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from algorithms import GD, SGD
from utils import add_intercept, compute_accuracy, empirical_risk


def main() -> None:
    # load data
    folder = "data"
    data = np.load(os.path.join(folder, "inputs.npy"))
    labels = np.load(os.path.join(folder, "labels.npy"))
    # get info about data
    n = data.shape[0]
    d = data.shape[1]
    print(f"n: {n}")
    print(f"d: {d}")

    # preprocess the data
    # center the data
    data = data - data.mean(axis=0)
    # standardize the data
    data = data / data.std(axis=0)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.33, random_state=2
    )
    n_train = X_train.shape[0]


    """
        Scikit
    """
    print("scikit")
    tic = time()
    clf = LogisticRegression(random_state=0).fit(X_train, y_train.reshape(n_train))
    toc = time()
    scikit_time = toc - tic
    theta_scikit = np.append(clf.coef_[0], clf.intercept_)
    scikit_test_accuracy = clf.score(X_test, y_test)
    scikit_train_accuracy = clf.score(X_train, y_train)

    # scikit returns a theta that contains the intercept
    X_train = add_intercept(X_train)
    X_test = add_intercept(X_test)
    scikit_empirical_risk = empirical_risk(theta_scikit, X_train, y_train)
    print(f"scikit loss: {scikit_empirical_risk}")


    """
        GD
    """
    # set the algorithm
    mu = 10
    gamma_GD = 10
    max_n_iterations_gd = int(1e2)
    # run algorithm
    tic = time()
    theta_gd = GD(
        gamma_GD,
        X_train,
        y_train,
        max_n_iterations_gd,
        scikit_empirical_risk,
        mu,
    )
    toc = time()
    gd_time = toc - tic

    """
        SGD
    """
    # set up the algorithm
    gamma_0 = 10
    max_n_iterations_sgd = int(1e5)
    schedule = "constant"
    tic = time()
    # run the algorithm
    theta_sgd = SGD(
        gamma_0,
        X_train,
        y_train,
        max_n_iterations_sgd,
        scikit_empirical_risk,
        mu,
        schedule,
    )
    toc = time()
    sgd_time = toc - tic


    print(f"\nscikit accuracy train: {scikit_train_accuracy:.2f}")
    print(f"GD accuracy train: {compute_accuracy(theta_gd, X_train, y_train):.2f}")
    print(f"SGD accuracy train: {compute_accuracy(theta_sgd, X_train, y_train):.2f}")

    print(f"\nscikit accuracy test: {scikit_test_accuracy:.2f}")
    print(f"GD accuracy test: {compute_accuracy(theta_gd, X_test, y_test):.2f}")
    print(f"SGD accuracy test: {compute_accuracy(theta_sgd, X_test, y_test):.2f}")

    print(f"\nscikit time: {scikit_time:.4f} seconds")
    print(f"GD time: {gd_time:.4f} seconds")
    print(f"SGD time: {sgd_time:.4f} seconds ")


if __name__ == "__main__":
    main()
