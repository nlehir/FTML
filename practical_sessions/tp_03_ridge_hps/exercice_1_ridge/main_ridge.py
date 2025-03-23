"""
    Study the statistical properties of the Ridge estimator:
        - excess risk
        - dependence of the risk on the dimensions n and d

    and compare those to the same quantities for OLS, in order to
    witness the benefits of using Ridge regression.
"""

import numpy as np

# from utils_algo import ridge_test_error
from utils_algo_solution import ridge_test_error
from utils_plots import plot_test_errors_ridge


def main() -> None:
    # dimensions of the problem
    n_train = 40
    d_list = [20, 40]
    d_list = [10, 20, 30, 40]
    # d_list = [40]
    # number of tests to estimate the excess risk
    # n_repetitions_list = [10, 100, 1000, 10000, 100000]
    # n_repetitions_list = [10000]
    n_repetitions_list = [100000]

    exponents = np.arange(-5, 5, 0.5)
    lambda_list = [10 ** (u) for u in exponents]

    theta_star_type_list = ["random", "eigenvalue_largest", "eigenvalue_smallest"]
    # design_matrix_type_list = ["uniform", "low_rank"]
    design_matrix_type_list = ["uniform"]

    # Assess the influence of different values of n and d
    for design_matrix_type in design_matrix_type_list:
        print(f"\n{design_matrix_type} design matrix")
        for theta_star_type in theta_star_type_list:
            print(f"{theta_star_type} theta")
            for n_repetitions in n_repetitions_list:
                test_errors = dict()
                print(f"{n_repetitions} repetitions")
                for d in d_list:
                    # print(f"d: {d}")
                    for lambda_ in lambda_list:
                        # print(f"lambda: {lambda_}")
                        test_errors[(d, lambda_)] = ridge_test_error(
                            n_train=n_train,
                            d=d,
                            lambda_=lambda_,
                            n_repetitions=n_repetitions,
                            theta_star_type=theta_star_type,
                            design_matrix_type=design_matrix_type,
                        )
                plot_test_errors_ridge(
                    risks=test_errors,
                    d_list=d_list,
                    n=n_train,
                    lambda_list=lambda_list,
                    n_repetitions=n_repetitions,
                    theta_star_type=theta_star_type,
                    design_matrix_type=design_matrix_type,
                )


if __name__ == "__main__":
    main()
