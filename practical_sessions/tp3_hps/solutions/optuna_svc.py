"""
Perform HP optimization with Optuna.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVC

from params import n_splits, test_size

digits = load_digits()

X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)


def objective(trial) -> float:
    """
    Objective function

    This function return the test mean accuracy
    after fitting a ridge classifier with a given set of hyperparameters.

    Fix this function by using the optuna API.
    https://optuna.org/
    """
    # split the data into training and test

    # suggest a alpha
    C = trial.suggest_float("C", 1e-10, 5e2)
    rbf_gamma = trial.suggest_float("rbf_gamma", 1e-7, 1e3)
    # poly_degree = trial.suggest_int("degree", 1, 8)
    shrinking_int = trial.suggest_int("shrinking", 0, 1)
    if shrinking_int:
        shrinking = True
    else:
        shrinking = False
    kernels = ["linear", "sigmoid", "poly", "rbf"]
    kernel = trial.suggest_categorical("kernel", kernels)

    # instantiate Ridge regressor object
    classifier = SVC(C=C, kernel=kernel, shrinking=shrinking, gamma=rbf_gamma)

    kf = KFold(n_splits=n_splits)
    scores = cross_val_score(classifier, X_train, y_train, cv=kf)
    cross_validation_score = scores.mean()
    return cross_validation_score


def main() -> None:
    storage_name = "svc.db"
    # clean study if exists
    # you may not want this behavior in practical applications
    if os.path.exists(storage_name):
        os.remove(storage_name)
    # create a study
    study = optuna.create_study(
        storage=f"sqlite:///{storage_name}",
        study_name="svc",
        load_if_exists=False,
        direction="maximize",  # we want to maximize the R2 score
    )
    study.optimize(objective, n_trials=100)

    # print best trial
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    for key, value in study.best_trial.params.items():
        if type(value) == float:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # extract dataframe
    df = study.trials_dataframe()

    # boxplot with kernel
    sns.boxplot(data=df, x="params_kernel", y="value")
    plt.title("influence of the kernel on the final mean accuracy")
    plt.savefig("boxplot_kernel.pdf")


if __name__ == "__main__":
    main()
