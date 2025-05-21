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
    X: np.ndarray,
    wh: np.ndarray,
    theta: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Compute the forward pass of a neural network with an output dimension
    of 1, and with only one hidden layer of m neurons.

    Also returns the intermediate results, that are useful for gradient computations.
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

    """
    FIX THIS FUNCTION
    """
    # stack X with a column of 1s in order to add the intercepts
    ones_X = np.ones(shape=(n, 1))
    X_stacked = np.column_stack((X, ones_X))

    # linear product between inputs and first hidden layer
    pre_h = np.ones((n, wh.shape[1]))

    # apply non linearity
    h = pre_h

    # stack h with a column of 1s in order to add the intercepts
    ones_h = np.ones(shape=(n, 1))
    h_stacked = np.column_stack((h, ones_h))

    # linear operation between hidden layer and output layer
    pre_y = 1

    # apply non linearity
    y_hat = np.ones((n, 1))

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

    FIX these gradients !
    """
    # first compute the gradient with respect to theta
    dl_dy_hat = y_hat - y

    # return
    gradients = dict()
    gradients["dl_dtheta"] = theta
    gradients["dl_dwh"] = np.ones((2, len(theta) - 1))
    return gradients
