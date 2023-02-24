"""
fit the noisy data
"""

import matplotlib.pyplot as plt
import csv
import ipdb
import numpy as np
import random

# open file
file_name = 'noisy_data.csv'

inputs = []
outputs = []

with open(file_name, 'r') as f:
    reader = csv.reader(f)
    # read file row by row
    # convert to lists
    row_index = 0
    for row in reader:
        # print(row)
        # if row_index >= 1:
        if True:
            inputs.append(float(row[0]))
            outputs.append(float(row[1]))
        row_index = row_index + 1

# print(inputs)
# print(outputs)
xlim_left = min(inputs)
xlim_right = max(inputs)
ylim_top = max(outputs)
ylim_bottom = min(outputs)

"""
randomly select training set and test set from the dataset
"""

nb_points = len(inputs)
nb_training_points = int(0.7 * nb_points)
training_indexes = random.sample(range(nb_points), nb_training_points)
test_indexes = [index for index in range(nb_points)
                if index not in training_indexes]
# print(training_indexes)
# print(test_indexes)

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


def compute_test_error(polynom, x_test, y_test):
    """
        Evaluate the quality of out model on the test set.
        We compute the Mean Square Error.
    """
    # compare prediction to ground truth
    errors = np.polyval(polynom, x_test) - y_test
    square_errors = [error**2 for error in errors]
    total_error = sum(square_errors)
    mean_square_error = total_error/len(square_errors)
    return mean_square_error


def compute_training_error(polynom, x_train, y_train):
    """
        Evaluate the quality of out model on the training set.
        We compute the Mean Square Error.
    """
    # compare prediction to ground truth
    errors = np.polyval(polynom, x_train) - y_train
    square_errors = [error**2 for error in errors]
    total_error = sum(square_errors)
    mean_square_error = total_error/len(square_errors)
    return mean_square_error


def plot_polynom_sample(polynom, x_train, y_train):
    """
        Plot the result of fitting the polynom
        to the training set
    """
    degree = len(polynom)-1
    title = f"Polynomial fit on training set, degree={degree}"
    filename = f"Fit_degree_{degree}.pdf"
    x_plot = np.linspace(xlim_left, xlim_right, 500)
    plt.plot(x_train,
             y_train,
             'o',
             x_test,
             y_test,
             'x',
             x_plot,
             np.polyval(polynom, x_plot), '-')
    plt.legend(['training set', 'test set', 'model'], loc='best')
    plt.xlabel('training inputs')
    plt.ylabel('training outputs')
    plt.xlim(xlim_left, xlim_right)
    plt.ylim(ylim_bottom, ylim_top)
    plt.title(title)
    plt.savefig('images/' + filename)
    plt.close()


def plot_polynom_zoom_out(polynom, x_train, y_train):
    """
       Plot the result of fitting the polynom
       and its predictions on points that are
       far from the training set.
    """
    degree = len(polynom)-1
    title = f"Polynom prediction on global dataset, degree={degree}"
    filename = f"Global_degree_{degree}.pdf"
    x_out_left = -200
    x_out_right = 200
    x_plot = np.linspace(x_out_left, x_out_right, 500)
    plt.plot(inputs, outputs, 'o', x_plot, np.polyval(polynom, x_plot), '-')
    plt.xlabel('global inputs')
    plt.ylabel('global outputs')
    plt.xlim(x_out_left, x_out_right)
    plt.ylim(-20000, 20000)
    plt.title(title)
    plt.savefig('images/' + filename)
    plt.close()


for degree in range(35):
    print(f"---\npolynom degree {degree}")
    poly = fit_polynom(degree, x_train, y_train)
    print(
        f"mean square error on training set: {compute_training_error(poly, x_train, y_train)}")
    print(
        f"mean square error on test set: {compute_test_error(poly, x_test, y_test)}")
    plot_polynom_sample(poly, x_train, y_train)
    plot_polynom_zoom_out(poly, x_train, y_train)
