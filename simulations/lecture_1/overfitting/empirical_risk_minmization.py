"""
Overfitting demo.

fit the noisy data by empirical risk minimization,
using polynomial functions of various degrees.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
from oracle import oracle, sigma

# open file
file_name = "noisy_data.csv"
data = np.genfromtxt(file_name, delimiter=",")
inputs = data[:, 0]
outputs = data[:, 1]


xlim_left = -0.1
xlim_right = 1.1
ylim_top = 1.3 * max(outputs)
ylim_bottom = 1.3 * min(outputs)

"""
randomly select training set and test set from the dataset
"""
nb_points = len(inputs)
nb_training_points = int(0.7 * nb_points)
training_indexes = random.sample(range(nb_points), nb_training_points)
test_indexes = [index for index in range(nb_points) if index not in training_indexes]

x_train = [inputs[i] for i in training_indexes]
y_train = [outputs[i] for i in training_indexes]
x_test = [inputs[i] for i in test_indexes]
y_test = [outputs[i] for i in test_indexes]


def fit_polynom(degree, x_train, y_train):
    """
    We use numpy's method in order to fit polynoms
    to the data
    """
    return np.polyfit(x_train, y_train, degree)


def compute_error(polynom, x, y):
    """
    Evaluate the quality of out model on the test set.
    We compute the Mean Square Error.
    """
    errors = np.polyval(polynom, x) - y
    square_errors = [error**2 for error in errors]
    total_error = sum(square_errors)
    mean_square_error = total_error / len(square_errors)
    return mean_square_error


def plot_polynom_sample(polynom, x_train, y_train, test_error, train_error):
    """
    Plot the result of fitting the polynom
    to the training set
    """
    degree = len(polynom) - 1
    title = f"Polynomial fit on training set, degree={degree}\ntrain error {train_error:.2E}, test error {test_error:.2E}"
    filename = f"Fit_degree_{degree}.pdf"
    x_plot = np.linspace(xlim_left, xlim_right, 500)
    # training set
    plt.plot(x_train, y_train, "o", markersize=3, alpha=0.8, label="training set")
    # test set
    plt.plot(x_test, y_test, "x", markersize=3, alpha=0.8, label="test set")
    # plot empirical risk minimizer
    plt.plot(
        x_plot,
        np.polyval(polynom, x_plot),
        "-",
        markersize=3,
        alpha=0.5,
        label="Empirical risk minimizer",
    )
    plt.plot(x_plot, oracle(x_plot), color="aqua", alpha=0.5, label="Bayes predictor")
    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(xlim_left, xlim_right)
    plt.ylim(ylim_bottom, ylim_top)
    plt.title(title)
    file_path = os.path.join("images", filename)
    plt.savefig(file_path)
    plt.close()


max_degree = 20
test_errors = list()
train_errors = list()
for degree in range(max_degree):
    polynom = fit_polynom(degree, x_train, y_train)
    test_error = compute_error(polynom, x_test, y_test)
    train_error = compute_error(polynom, x_train, y_train)
    test_errors.append(test_error)
    train_errors.append(train_error)
    plot_polynom_sample(polynom, x_train, y_train, test_error, train_error)
    print(f"---\npolynom degree {degree}")
    print(f"mean square error on training set: {train_error:.2E}")
    print(f"mean square error on test set: {test_error:.2E}")


# plot test and train errors
plt.plot(range(max_degree), train_errors, "o", label="train error", markersize=3)
plt.plot(range(max_degree), test_errors, "x", label="test error", markersize=3)
plt.plot(
    range(max_degree),
    np.ones(max_degree) * sigma**2,
    label="Bayes risk",
    color="aqua",
)
plt.xlabel("polynom degree")
plt.legend(loc="best")
plt.title("underfitting and overfitting")
plt.savefig("images/overfitting.pdf")
