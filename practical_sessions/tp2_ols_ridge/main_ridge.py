"""
    Study the statistical properties of the OLS estimator:
        - excess risk
        - dependence of the risk on the dimensions n and d
        - (optionally) stability of the OLS estimator
"""
from utils_algo import  ridge_risk
from utils_plots import plot_test_errors_ridge


def main() -> None:
    # dimensions of the problem
    # n_list = list(range(30, 1000, 100))
    n = 30
    # d_list = list(range(0, 1000, 200))
    # d_list.remove(0)
    # d_list = [2, 10, 50]
    d_list = [10, 20, 30, 40]

    exponents = [k for k in range(-6, 6)]
    lambda_list = [10**(u) for u in exponents]

    # number of tests to estimate the excess risk
    n_tests = int(1e4)

    # Assess the influence of different values of n and d
    test_errors = dict()
    for d in d_list:
        print(f"d: {d}")
        for lambda_ in lambda_list:
            # print(f"lambda: {lambda_}")
            test_errors[(d, lambda_)] = ridge_risk(n, d, lambda_, n_tests)
    plot_test_errors_ridge(test_errors, d_list, n, lambda_list)


if __name__ == "__main__":
    main()
