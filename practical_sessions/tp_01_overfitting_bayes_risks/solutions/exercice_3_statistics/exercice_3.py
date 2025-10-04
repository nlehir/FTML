"""
Study overfitting and variance of the test error estimation
by monitoring the R2 train and test scores after subsampling the datasets
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def downsample_dataset(X, y, n, rng):
    test_indexes = rng.choice(np.arange(len(X)), size=n, replace=False)
    X_downsampled = X[test_indexes]
    y_downsampled = y[test_indexes]
    return X_downsampled, y_downsampled


def test_error_empirical_std(n_test, estimator, rng, X_test, y_test):
    n_tests_to_compute_variance = 1000
    test_scores = list()
    for _ in range(n_tests_to_compute_variance):
        X_test_subsampled, y_test_subsampled = downsample_dataset(
            X=X_test,
            y=y_test,
            n=n_test,
            rng=rng,
        )
        test_scores.append(estimator.score(X_test_subsampled, y_test_subsampled))
    std = np.asarray(test_scores).std()
    return std


def study_test_error_empirical_std(estimator, rng, X_test, y_test, n_train):
    print("Study test error std")
    n_test_list = np.arange(10, 1000, 10)
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
    linear_reg_std_log = LinearRegression()
    log_n_test = np.log10(n_test_list).reshape(len(n_test_list), 1)
    log_std = np.log10(std_list)
    linear_reg_std_log.fit(X=log_n_test, y=log_std)
    # x_linear
    y_pred_linear = linear_reg_std_log.predict(log_n_test)
    plt.plot(log_n_test, log_std, "o", alpha=0.7)
    text = f"linear regression on logs: {linear_reg_std_log.coef_.item():.2f}"
    plt.plot(log_n_test, y_pred_linear, label=text)
    plt.legend(loc="best")
    plt.xlabel("log10(n test)")
    plt.ylabel("log10 of empirical standard deviation of the test error")
    # plt.yscale("log")
    # plt.xscale("log")
    title = f"Empirical standard deviation of the test error\nn_train: {n_train}"
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
        X_train_subsampled, y_train_subsampled = downsample_dataset(
            X=X_train,
            y=y_train,
            n=n_train,
            rng=rng,
        )
        estimator = LinearRegression()
        estimator.fit(X_train_subsampled, y_train_subsampled)
        train_score = estimator.score(X_train_subsampled, y_train_subsampled)
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
    n_train = 200
    X_train_subsampled, y_train_subsampled = downsample_dataset(
        X=X_train,
        y=y_train,
        n=n_train,
        rng=rng,
    )
    estimator = LinearRegression()
    estimator.fit(X_train_subsampled, y_train_subsampled)
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
