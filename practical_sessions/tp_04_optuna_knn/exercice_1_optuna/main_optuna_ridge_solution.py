import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    train_test_split,
)

# load data
X_path = os.path.join("data_ridge", "X.npy")
y_path = os.path.join("data_ridge", "y.npy")
X = np.load(X_path)
y = np.load(y_path)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

def objective(trial) -> float:
    """
    Objective function

    Return the r2 score on the validation set,
    after fitting a ridge estimator with a given set of hyperparameters.
    """
    alpha = trial.suggest_float("alpha", 1e-12, 1e2)

    # solvers
    available_solvers = ["cholesky", "lsqr", "svd", "sag"]
    solver = trial.suggest_categorical("solver", available_solvers)

    # load estimator object

    estimator = Ridge(alpha=alpha, solver=solver)
    kf = KFold(n_splits=5)
    scores = cross_val_score(estimator, X_train, y_train, cv=kf)

    # return validation score
    return scores.mean()


def prediction_squared_error(estimator, X, y):
    predictions = estimator.predict(X)
    n_samples = X.shape[0]
    error = predictions - y
    return np.linalg.norm(error) ** 2 / n_samples


def main():
    # database for the optuna dashboard
    storage_name = "ridge.db"
    # clean study if exists
    # you may not want this behavior in practical applications
    if os.path.exists(storage_name):
        os.remove(storage_name)
    # create a study
    study = optuna.create_study(
        storage=f"sqlite:///{storage_name}",
        study_name="Ridge_regression",
        load_if_exists=False,
        direction="maximize",  # we want to maximize the R2 score
    )
    study.optimize(func=objective, n_trials=50)

    # print best trial
    print(f"\nBest cross validation score: {study.best_value:.3f} (params: {study.best_params})")
    for key, value in study.best_trial.params.items():
        if type(value) == float:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # extract dataframe
    df = study.trials_dataframe()

    best_alpha = study.best_params["alpha"]
    estimator = Ridge(alpha=best_alpha)
    print(f"Train ridge with alpha {best_alpha:.3f}")
    estimator.fit(X=X_train, y=y_train)
    test_r2 = estimator.score(X=X_test, y=y_test)
    print(f"Final test r2: {test_r2:.3f}")


    # analyze the hyperparameters
    sns.boxplot(data=df, x="params_solver", y="value")
    title = (
        "influence of the solver on the validation R2 score"
        f"\ntest R2: {test_r2:.3f}"
            )
    plt.title(title)
    plt.ylabel("Cross validation r2")
    figpath = os.path.join("images", "solver.pdf")
    plt.savefig(figpath)
    plt.close()

    plt.plot(df.params_alpha, df.value, "o")
    title = (
            "Influence of the regularization parameter on the validation R2 score"
            f"\ntest R2: {test_r2:.3f}"
            )
    plt.title(title)
    plt.xlabel("regularization constant")
    plt.ylabel("Cross validation r2")
    figpath = os.path.join("images", "regularization.pdf")
    plt.savefig(figpath)
    plt.close()


if __name__ == "__main__":
    main()
