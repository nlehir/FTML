"""
    Study the statistical properties of the OLS estimator:
        - excess risk
        - dependence of the risk on the dimensions n and d
        - (optionally) stability of the OLS estimator
"""


# from utils_algo_solution import ols_test_error
from utils_algo import ols_test_error
from utils_plots import plot_test_errors_ols


def main() -> None:
    # dimensions of the problem
    n_list = list(range(30, 200, 2))
    d_list = [2, 5, 10, 20, 30]


    # Assess the influence of different values of n and d
    # store the test errors in a dict, each key being a (n, d) pair
    test_errors = dict()
    for n in n_list:
        for d in d_list:
            test_errors[(n, d)] = ols_test_error(n=n, d=d)
    plot_test_errors_ols(
            test_errors=test_errors,
            n_list=n_list,
            d_list=d_list,
            )


if __name__ == "__main__":
    main()
