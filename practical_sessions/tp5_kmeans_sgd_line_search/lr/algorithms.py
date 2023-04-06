import numpy as np

from utils import empirical_risk, sigmoid

from learning_rate_schedule import learning_rate_schedule

conv_tolerance = 1.05


def gradient_estimate(theta: np.ndarray, x: np.ndarray, y: float, mu) -> np.ndarray:
    """
    Gradient estimate, used by SGD.
    Here, like in many ML applications, the estimate is the gradient
    with respect to one sample of the dataset only (instead of the batch
    gradient).

    x: vector of length d: represents the input
    theta: vector of length d: represents the linear separator
    y: -1 or 1, represents de class
    """
    d = len(theta)
    grad_estimate = -(x * y * sigmoid(-x.dot(theta) * y)).reshape(d, 1) + mu * theta
    return grad_estimate


def batch_gradient(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, mu: float
) -> np.ndarray:
    """
    Batch gradient for logistic regression.

    X: design matrix
    y: labels

    Uses elementwise vector products (numpy syntax).
    """
    n = X.shape[0]
    products = (X @ theta) * y
    gradient = 1 / n * (X.T) @ (-y * sigmoid(-products)) + mu * theta
    return gradient


def GD(
    gamma,
    X_train,
    y_train,
    n_iterations,
    scikit_empirical_risk,
    mu,
) -> np.ndarray:
    """
    Vanilla gradient descent
    """
    print("\nGD")
    d = X_train.shape[1]
    theta_gd = np.zeros((d, 1))

    # algorithm
    for iteration in range(n_iterations):
        # compute only some empirical risks in order to save time
        if iteration % (n_iterations // 10) == 0:
            empirical_risk_iter = empirical_risk(theta_gd, X_train, y_train)
            print(f"iteration: {iteration}/{n_iterations}")
            print(f"loss: {empirical_risk_iter:.3f}")
            # early stopping
            if empirical_risk_iter < conv_tolerance * scikit_empirical_risk:
                print(
                    f"attained {conv_tolerance} X scikit risk in {iteration} iterations"
                )
                print("exit GD")
                break
        theta_gd -= gamma * batch_gradient(theta_gd, X_train, y_train, mu)
    return theta_gd


def SGD(
    gamma_0,
    X_train,
    y_train,
    n_iterations,
    scikit_empirical_risk,
    mu,
    schedule,
) -> np.ndarray:
    print("\nSGD")
    d = X_train.shape[1]
    theta_sgd = np.zeros((d, 1))
    n_train = X_train.shape[0]
    rng = np.random.default_rng()
    for iteration in range(n_iterations):
        if iteration % (n_iterations // 10) == 0:
            empirical_risk_iter = empirical_risk(theta_sgd, X_train, y_train)
            print(f"iteration: {iteration}/{n_iterations}")
            print(f"loss: {empirical_risk_iter:.3f}")
            # early stopping
            if empirical_risk_iter < conv_tolerance * scikit_empirical_risk:
                print(
                    f"attained {conv_tolerance} X scikit risk in {iteration} iterations"
                )
                print("exit SGD")
                break

        # choose the learning rate gamma
        gamma = learning_rate_schedule(gamma_0, iteration, schedule)

        # uniformly sample an index in the dataset
        index = rng.integers(n_train)
        x = X_train[index]
        y = y_train[index]
        # compute the gradient estimate
        theta_sgd -= gamma * gradient_estimate(theta_sgd, x, y, mu)
    return theta_sgd
