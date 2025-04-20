import numpy as np


def gamma_line_search(H: np.ndarray, gradient: np.ndarray) -> float:
    """
    Exact line search gamma

    Edit ths function
    """
    return 1


def line_search(
    X,
    H,
    y,
    theta_hat,
    n_iterations,
    ax_linear,
    ax_log,
):
    """
    Perform exact line search gradient descent

    Edit ths function
    """
    label = "line search"
    x_plot = range(1, n_iterations + 1)
    LS_squared_distances_to_opt = [1 for x in x_plot]
    ax_linear.plot(x_plot, LS_squared_distances_to_opt, label=label)
    ax_log.plot(x_plot, LS_squared_distances_to_opt, label=label)


def gradient_descent(
    X,
    H,
    y,
    theta_hat,
    gamma,
    n_iterations,
    ax_linear,
    ax_log,
):
    """
    Perform vanilla gradient descent

    Edit ths function
    """
    label = r"$\gamma=$" f"{gamma}"
    x_plot = range(1, n_iterations + 1)
    GD_squared_distances_to_opt = [1 for x in x_plot]
    ax_linear.plot(x_plot, GD_squared_distances_to_opt, label=label)
    ax_log.plot(x_plot, GD_squared_distances_to_opt, label=label)


def OLS_estimator(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute OLS estimators from the data.

    We use numpy broadcasting to accelerate computations
    and obtain several OLS estimators.

    Parameters:
        X: (n, d) matrix
        y: (n, n_tests) matrix

    Returns:
        theta_hat: (d, n_tests) matrix
    """
    covariance_matrix = X.T @ X
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = inverse_covariance @ (X.T @ y)
    return theta_hat


def gradient(theta, H, X, y):
    """
    Compute the gradient of the empirical risk
    as a function of theta, X, y
    for a least squares problem.

    Parameters:
        X (float matrix): (n, d) matrix
        y (float vector): (n, 1) vector
        theta (float vector): (d, 1) vector

    Returns:
        gradient of the objective function

    Edit this function.
    """
    return theta
