"""
    Utilities for application of SGD on the neural network.
    Fix this file.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU non linearity
    """
    return np.maximum(x, 0)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the ReLU non linearity
    """
    return np.heaviside(x, 0)


def forward_pass(
    X: np.ndarray, wh: np.ndarray, theta: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Compute the forward pass of a neural network with an output dimension
    of 1, and with only one hidden layer of m neurons.

    Also return the intermediate results, that are useful for gradient computations.
    A dict is used to avoid dependence on the order of the returned variables.

    X:     (n, d) array
    (inputs: n inputs in dimension d)
    In this exercice, d=1
    Each input is thus a line vector.

    wh:    (d+1, m) array
    (weights between the input layer and the hidden layer)

    theta: (m+1, 1) array
    (weights between the hidden layer and the output)
    """

    if X.shape:
        n = X.shape[0]
    else:
        n = 1

    # stack X with a column of 1s in order to add the intercepts
    ones_X = np.ones(shape=(n, 1))
    X_stacked = np.column_stack((X, ones_X))

    # linear product between inputs and first hidden layer
    pre_h = X_stacked @ wh

    # apply non linearity
    h = relu(pre_h)

    # stack h with a column of 1s in order to add the intercepts
    ones_h = np.ones(shape=(n, 1))
    h_bar = np.column_stack((h, ones_h))

    # linear operation between hidden layer and output layer
    pre_y = h_bar @ theta

    # apply non linearity
    y_hat = relu(pre_y)

    # return all the steps (useful for gradients)
    outputs = dict()
    outputs["pre_h"] = pre_h
    outputs["h"] = h
    outputs["pre_y"] = pre_y
    outputs["y_hat"] = y_hat
    return outputs


def compute_gradients(
    x: np.ndarray,
    y: np.ndarray,
    pre_h: np.ndarray,
    h: np.ndarray,
    pre_y: np.ndarray,
    y_hat: np.ndarray,
    theta: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    The gradient makes use of several intermediate
    variables returned by the forward pass, see the
    explanations in the pdf for mode details and for
    the details of the calculations.

    Since we use a SGD, we only compute the gradient
    with respect to 1 sample.

    l is the squared loss.

    We use the chain rule to write the computation.
    """
    # first compute the gradient with respect to theta
    dl_dy_hat = y_hat - y

    dy_hat_dpre_y = relu_derivative(pre_y)
    # dy_hat_dpre_y = pre_y

    h_bar = np.append(h, 1)
    dpre_y_dtheta = h_bar
    dl_dtheta = dl_dy_hat * dy_hat_dpre_y * dpre_y_dtheta

    # then compute the gradient with respect to w_h
    # see pdf "Gradients.pdf" for details
    theta_tilde = theta[:-1]
    x_bar = np.append(x, 1)
    x_bar = x_bar.reshape(1, len(x_bar))
    u = theta_tilde * relu_derivative(pre_h)
    dl_dwh = (dl_dy_hat * dy_hat_dpre_y) * u.T @ x_bar

    # transpose the jacobians to obtain the gradients
    gradients = dict()
    gradients["dl_dtheta"] = dl_dtheta.T
    gradients["dl_dwh"] = dl_dwh.T
    return gradients
