"""
Study overfitting and variance of the test error estimation
by monitoring the R2 train and test scores after subsampling the datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression


def test_error_empirical_std(n_test, estimator, rng, X_test, y_test):
    """
    TODO: same computation without a for loop ?
    """
    n_tests_to_compute_variance = 150
    test_scores = list()
    for _ in range(n_tests_to_compute_variance):
        test_indexes = rng.choice(np.arange(len(X_test)), size=n_test)
        X_test_subsampled = X_test[test_indexes]
        y_test_subsampled = y_test[test_indexes]
        test_scores.append(estimator.score(X_test_subsampled, y_test_subsampled))
    std = np.asarray(test_scores).std()
    return std


def study_test_error_empirical_std(estimator, rng, X_test, y_test, n_train):
    print("Study test error std")
    n_test_list = np.arange(10, 500, 10)
    std_list = list()
    for n_test in n_test_list:
        print(f"{n_test=}")
        std_list.append(
            test_error_empirical_std(
                n_test=n_test,
                estimator=estimator,
                rng=rng,
                X_test=X_test,
                y_test=y_test,
            )
        )
    plt.plot(n_test_list, std_list, "o", alpha=0.7)
    plt.xlabel("n test")
    plt.ylabel("empirical standard deviation of the test error")
    title = "Empirical standard deviation of the test error\n" f"n_train: {n_train}"
    plt.title(title)
    plt.savefig(f"std_{n_train=}.pdf")
    plt.close()


def study_overfitting(
    X_train,
    X_test,
    y_train,
    y_test,
    rng,
):
    print("Study overfitting")
    n_train_list = np.arange(10, 500, 10)
    overfitting_list = list()
    n_test = len(X_test)
    for n_train in n_train_list:
        print(f"{n_train=}")
        train_indexes = rng.choice(np.arange(len(X_train)), size=n_train)
        X_train_subsample = X_train[train_indexes]
        y_train_subsample = y_train[train_indexes]
        estimator = LinearRegression()
        estimator.fit(X_train_subsample, y_train_subsample)
        train_score = estimator.score(X_train_subsample, y_train_subsample)
        test_score = estimator.score(X_test, y_test)
        overfitting_list.append(train_score - test_score)
    plt.plot(n_train_list, overfitting_list, "o", alpha=0.7)
    plt.xlabel("n train")
    plt.ylabel("train R2 - test R2")
    plt.yscale("log")
    title = (
        "Overfitting amount (difference between R2 train and R2 test)"
        f"\nn_test: {n_test}"
    )
    plt.title(title)
    plt.savefig(f"overfitting.pdf")
    plt.close()


def main():
    X_train = np.load(os.path.join("data", "X_train.npy"))
    X_test = np.load(os.path.join("data", "X_test.npy"))
    y_train = np.load(os.path.join("data", "y_train.npy"))
    y_test = np.load(os.path.join("data", "y_test.npy"))

    rng = np.random.default_rng()

    """
    Study the variance of the test error estimation
    by subsampling the test set
    n_train will also have an influence on the result
    """
    n_train = 300
    X_train_subsample = rng.choice(X_train, size=n_train)
    y_train_subsample = rng.choice(y_train, size=n_train)
    estimator = LinearRegression()
    estimator.fit(X_train_subsample, y_train_subsample)
    study_test_error_empirical_std(
        estimator=estimator,
        rng=rng,
        X_test=X_test,
        y_test=y_test,
        n_train=n_train,
    )

    """
    Study overfitting
    by subsampling the train set
    """
    study_overfitting(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        rng=rng,
    )


if __name__ == "__main__":
    main()
