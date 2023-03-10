"""
    Study the statistical properties of the OLS estimator:
        - excess risk
        - dependence of the risk on the dimensions n and d
        - (optionally) stability of the OLS estimator
"""
from utils_algo import ols_risk
from utils_plots import plot_stds, plot_test_errors_ols


def main() -> None:
    # dimensions of the problem
    # n_list = list(range(30, 1000, 100))
    n_list = list(range(30, 51, 2))
    # d_list = list(range(0, 1000, 200))
    # d_list.remove(0)
    # d_list = [2, 10, 50]
    d_list = [2, 5, 10, 20, 30]

    # number of tests to estimate the excess risk
    n_tests = int(1e3)

    # Assess the influence of different values of n and d
    risks = dict()
    # relative distances are optional, see pdf
    # relative_distance_std = dict()
    for n in n_list:
        for d in d_list:
            risks[(n, d)], _ = ols_risk(n, d, n_tests)
    plot_test_errors_ols(risks, n_list, d_list)
    # plot_stds(relative_distance_std, n_list, d_list)


if __name__ == "__main__":
    main()
