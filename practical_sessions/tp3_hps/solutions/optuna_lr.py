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
    # suggest a alpha
    C = trial.suggest_float("C", 1e-10, 5e2)

    # suggest a penalty
    penaltys = ["l2", None]
    penalty = trial.suggest_categorical("penalty", penaltys)

    solvers = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
    solvers = ["sag", "saga", "lbfgs", "newton-cg"]
    solver = trial.suggest_categorical("solver", solvers)

    # instantiate Ridge regressor object
    classifier = LogisticRegression(C=C, solver=solver, penalty=penalty, max_iter=400)

    # classifier.fit(X_train, y_train)
    kf = KFold(n_splits=n_splits)
    scores = cross_val_score(classifier, X_train, y_train, cv=kf)
    cross_validation_score = scores.mean()
    return cross_validation_score


def prediction_squared_error(classifier, X, y):
    predictions = classifier.predict(X)
    n_samples = X.shape[0]
    error = predictions - y
    return np.linalg.norm(error) ** 2 / n_samples


def main() -> None:
    storage_name = "lr.db"
    # clean study if exists
    # you may not want this behavior in practical applications
    if os.path.exists(storage_name):
        os.remove(storage_name)
    # create a study
    study = optuna.create_study(
        storage=f"sqlite:///{storage_name}",
        study_name="logistic_regression",
        load_if_exists=False,
        direction="maximize",
    )
    study.optimize(objective, n_trials=200)

    # print best trial
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    for key, value in study.best_trial.params.items():
        if type(value) == float:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # extract dataframe
    df = study.trials_dataframe()

    categorical_params = ["solver", "penalty"]
    for param in categorical_params:
        # boxplot with solver
        sns.boxplot(data=df, x=f"params_{param}", y="value")
        title = (
            "Logistic regression cv score\n"
            f"influence of the {param} on the final mean accuracy"
        )
        plt.title(title)
        fig_path = os.path.join("images", f"lr_boxplot_{param}.pdf")
        plt.savefig(fig_path)
        plt.close()


if __name__ == "__main__":
    main()
