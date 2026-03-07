"""
Evaluate the quality of randomly drawn parameters for a
1 dimensional linear regression.

The function empirical_risk() must be completed in utils.py
"""

import numpy as np

# from utils_solution import empirical_risk
from constants import STD_NOISE
from mse import mse
from termcolor import colored
from utils import empirical_risk
from utils_files import load_data


def main() -> None:
    n_tests = 100

    X_train, X_test, y_train, y_test = load_data(std=STD_NOISE)

    rng = np.random.default_rng()

    best_train_error = 10e12
    best_theta = 0
    best_b = 0

    for test_id in range(n_tests):
        theta = rng.uniform(-100, 100)
        b = rng.uniform(-100, 100)
        train_error = empirical_risk(
            theta=theta,
            b=b,
            X=X_train,
            y=y_train,
        )
        """
        Used as a ground truth for the exercise
        """
        MSE_train = mse(
            X=X_train,
            theta=theta,
            b=b,
            y_true=y_train,
        )
        if train_error < best_train_error:
            best_train_error = train_error
            best_theta = theta
            best_b = b
        print(
            f"test:             {test_id}"
            f"\ntheta:            {theta:.2f}"
            f"\nb:                {b:.2f}"
            f"\ntrain empirical risk:   {train_error:.2E}"
            f"\ntrain empirical risk (ground truth):   {MSE_train:.2E}\n"
        )
        """
        Test whether the results are equal
        """
        if np.isclose(a=train_error, b=MSE_train):
            message = "results are similar"
            color = "blue"
        else:
            message = "results are different"
            color = "yellow"
        print(colored(message, color=color))
    test_error = empirical_risk(
        theta=best_theta,
        b=best_b,
        X=X_test,
        y=y_test,
    )
    """
    Used as a ground truth for the exercise
    """
    MSE_test = mse(
        X=X_test,
        theta=best_theta,
        b=best_b,
        y_true=y_test,
    )
    print(
        "\n--------"
        f"\nbest theta:          {best_theta:.2f}"
        f"\nbest b:              {best_b:2f}"
        f"\nbest train empirical risk: {best_train_error:.2E}\n"
        f"\nbest test empirical risk: {test_error:.2E}"
        f"\nbest test empirical risk (ground truth): {MSE_test:.2E}\n"
    )
    """
    Test whether the results are equal
    """
    if np.isclose(a=test_error, b=MSE_test):
        message = "results are similar"
        color = "blue"
    else:
        message = "results are different"
        color = "yellow"
    print(colored(message, color=color))


if __name__ == "__main__":
    main()
