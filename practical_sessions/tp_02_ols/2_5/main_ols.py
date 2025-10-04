"""
Study the statistical properties of the OLS estimator:
    - excess risk
    - dependence of the risk on the dimensions n and d
    - (optionally) stability of the OLS estimator
"""

from utils_algo_solution import ols_test_error

# from utils_algo import ols_test_error
from utils_plots import plot_test_errors_ols


def main() -> None:
    # dimensions of the problem
    n_train_list = list(range(30, 200, 2))
    d_list = [2, 5, 10, 20, 30]
    n_repetitions_list = [10, 100, 1000, 10000]

    # Assess the influence of different values of n and d
    # store the test errors in a dict, each key being a (n, d) pair
    test_errors = dict()
    for n_repetitions in n_repetitions_list:
        print(f"{n_repetitions=}")
        for n_train in n_train_list:
            for d in d_list:
                test_errors[(n_train, d)] = ols_test_error(
                    n_train=n_train,
                    d=d,
                    n_repetitions=n_repetitions,
                )
        plot_test_errors_ols(
            test_errors=test_errors,
            n_list=n_train_list,
            d_list=d_list,
            n_repetitions=n_repetitions,
        )


if __name__ == "__main__":
    main()
