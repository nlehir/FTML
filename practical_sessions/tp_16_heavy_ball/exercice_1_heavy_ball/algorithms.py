import numpy as np
from time import time

ALPHA = 0.3

def gamma_line_search(H: np.ndarray, gradient: np.ndarray) -> float:
    """
    Exact line search gamma
    """
    square_norm = np.linalg.norm(gradient) ** 2
    d = gradient.shape[0]
    Hgrad = (H @ gradient).reshape(d)
    grad_reshape = gradient.reshape(d)
    inner_product = np.dot(Hgrad, grad_reshape)
    return square_norm / inner_product


def line_search(
    X,
    H,
    y,
    theta_hat,
    n_iterations,
    ax_linear,
    ax_log,
    tol,
    color,
):
    """
    Perform exact line search gradient descent
    """
    n, d = X.shape
    theta_0 = np.zeros((d, 1))
    LS_squared_distances_to_opt = list()
    theta_LS = theta_0.copy()

    tic = time()
    reached = False
    total_time = 0
    for iteration in range(1, n_iterations + 1):
        if iteration % (n_iterations//10)==0:
            print(f"iteration {iteration}/{n_iterations}")
        distance_to_opt = np.linalg.norm(theta_LS - theta_hat) ** 2
        LS_squared_distances_to_opt.append(distance_to_opt)
        grad = gradient(theta_LS, H, X, y)
        gamma_star = gamma_line_search(H, grad)
        theta_LS -= gamma_star * gradient(theta_LS, H, X, y)
        if distance_to_opt < tol:
            toc = time()
            reached = True
            print(f"iteration to reach distance of {tol}: {iteration}")
            total_time = toc - tic
            print(f"time to reach distance of {tol}: {total_time:.1E}")
            break

    final_iteration = len(LS_squared_distances_to_opt)
    if reached:
        label = f"Line search, {final_iteration} iters, {total_time:.1E} s"
        ax_linear.axvline(x=final_iteration, alpha=ALPHA, color=color)
        ax_log.axvline(x=final_iteration, alpha=ALPHA, color=color)
    else:
        label = "Line search"
    x_plot = range(1, final_iteration + 1)
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
    tol,
    color,
    theta_0,
    gamma_from_H,
):
    """
    Perform vanilla gradient descent
    """
    GD_squared_distances_to_opt = list()
    theta_GD = theta_0.copy()
    tic = time()
    reached = False
    total_time = 0
    for iteration in range(1, n_iterations + 1):
        if iteration % (n_iterations//10)==0:
            print(f"iteration {iteration}/{n_iterations}")
        distance_to_opt = np.linalg.norm(theta_GD - theta_hat) ** 2
        GD_squared_distances_to_opt.append(distance_to_opt)
        theta_GD -= gamma * gradient(theta_GD, H, X, y)
        if distance_to_opt < tol:
            toc = time()
            reached = True
            print(f"iteration to reach distance of {tol}: {iteration}")
            total_time = toc - tic
            print(f"time to reach distance of {tol}: {total_time:.1E}")
            break

    final_iteration = len(GD_squared_distances_to_opt)
    if reached:
        label = "GD " + r"$\gamma=$" f"{gamma:.2f}, {final_iteration} iters, {total_time:.1E} s"
        ax_linear.axvline(x=final_iteration, alpha=ALPHA, color=color)
        ax_log.axvline(x=final_iteration, alpha=ALPHA, color=color)
    else:
        label = "GD " + r"$\gamma=$" f"{gamma:.2f}"

    if gamma_from_H:
        label += f"=1/L"
    x_plot = range(1, final_iteration + 1)
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
    """
    n = y.shape[0]
    return H @ theta - 1 / n * X.T @ y

def upper_bound(
        theta_0,
        theta_hat,
        n_iterations,
        H,
        ):
    print("Compute upper bound")
    initial_distance_to_opt = np.linalg.norm(theta_0 - theta_hat) ** 2
    eigenvalues, _ = np.linalg.eig(H)
    condition_number = eigenvalues.max()/eigenvalues.min()
    upper_bounds = np.exp(-2*np.arange(n_iterations)/condition_number) * initial_distance_to_opt
    return upper_bounds

def heavy_ball(
    X,
    H,
    y,
    theta_hat,
    gamma,
    beta,
    n_iterations,
    ax_linear,
    ax_log,
    tol,
    color,
    theta_0,
):
    """
    Perform Heavy-Ball
    """
    print(f"\nHeavy ball, gamma={gamma}, beta={beta}")
    HB_squared_distances_to_opt = list()
    theta_HB = theta_0.copy()
    tic = time()
    reached = False
    total_time = 0
    theta_HB_before = theta_0.copy()
    for iteration in range(1, n_iterations + 1):
        if iteration % (n_iterations//10)==0:
            print(f"iteration {iteration}/{n_iterations}")
        distance_to_opt = np.linalg.norm(theta_HB - theta_hat) ** 2
        HB_squared_distances_to_opt.append(distance_to_opt)

        """
        Add heavy ball algorithm here
        """

        if distance_to_opt < tol:
            toc = time()
            reached = True
            print(f"iteration to reach distance of {tol}: {iteration}")
            total_time = toc - tic
            print(f"time to reach distance of {tol}: {total_time:.1E}")
            break

    final_iteration = len(HB_squared_distances_to_opt)
    if reached:
        # label = "Heavy ball " + r"$\gamma=$" f"{gamma:.2f}, {final_iteration} iters, {total_time:.1E} s"
        label = "Heavy ball " + f", {final_iteration} iters, {total_time:.1E} s"
        ax_linear.axvline(x=final_iteration, alpha=ALPHA, color=color)
        ax_log.axvline(x=final_iteration, alpha=ALPHA, color=color)
    else:
        # label = "Heavy ball " + r"$\gamma=$" f"{gamma:.2f}"
        label = "Heavy ball "

    x_plot = range(1, final_iteration + 1)
    ax_linear.plot(x_plot, HB_squared_distances_to_opt, label=label)
    ax_log.plot(x_plot, HB_squared_distances_to_opt, label=label)
