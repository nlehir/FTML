import os
import math
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return np.heaviside(x, 0)


def forward_pass(x, wh, theta):
    """
    wh: matrix of size (m, d+1) containing weights between
    the input layer and the hidden layer.

    theta: vector of size (m+1, 1) containing the weights
    between the hidden layer and the output.

    check the dimensions of the arrays !
    """
    pre_h = wh @ np.append(x, 1)
    h = relu(pre_h)
    pre_y = theta @ np.append(h, 1)
    y_hat = relu(pre_y)
    return pre_h, h, pre_y, y_hat


def gradients(x, y, pre_h, h, pre_y, y_hat, theta):
    """
    The gradient makes use of several intermediate
    variables returned by the forward pass, see the
    explanations in the pdf for mode details and for
    the details of the calculations.

    The Jacobian of a composition of functions is
    a product of jacobians.

    l is the squared los

    for instance, dl_dy_hat is the gradient
    of the loss with respect to y_hat (in this case, it is
    just a derivative).

    We use the chain rule to write the computation.
    """
    # first compute the gradient with respect to theta
    dl_dy_hat = y_hat - y
    dy_hat_dpre_y = relu_derivative(pre_y)
    dpre_y_dtheta = np.append(h, 1)
    dl_dtheta = dl_dy_hat * dy_hat_dpre_y * dpre_y_dtheta

    # then compute the gradient with respect to w_h
    # scalar
    dl_dpre_y = dl_dy_hat * dy_hat_dpre_y
    # vector with m components
    # we drop the last component of theta because it does not depend
    # on wh
    dpre_y_dh = theta[:-1]
    # vector with m components
    dl_dh = dl_dpre_y * dpre_y_dh
    # vector with m components
    dh_dpre_h = relu_derivative(pre_h)

    # vector with m components
    # this is an elementwise product
    dl_dpre_h = dl_dh * dh_dpre_h
    dl_dwh = np.matrix(dl_dpre_h).T * np.matrix(np.append(x, 1))
    return dl_dtheta, dl_dwh
