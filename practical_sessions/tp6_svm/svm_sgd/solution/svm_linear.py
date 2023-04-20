import os

import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored


def straight_line(a, b, x):
    return a * x + b


def side_of_line(point, separator_vector, offset):
    return np.dot(point, separator_vector) - offset


# load the data
dataset = "ex1"
data_1 = np.load("data/" + dataset + "/data_1.npy")
data_2 = np.load("data/" + dataset + "/data_2.npy")
if not os.path.exists(os.path.join("images/", dataset)):
    os.makedirs(os.path.join("images/", dataset))


def test_linear_separator(dataset, a, offset, data_1, data_2):
    # concatenate the data
    data = np.concatenate((data_1, data_2))

    # setup plot limits
    x_data_1 = data_1[:, 0]
    y_data_1 = data_1[:, 1]
    x_data_2 = data_2[:, 0]
    y_data_2 = data_2[:, 1]
    x_data = data[:, 0]
    y_data = data[:, 1]
    xlim_left = min(x_data)
    xlim_right = max(x_data)
    ylim_top = max(y_data)
    ylim_bottom = min(y_data)

    # test separator
    print(f"testing separator with a={a} and offset={offset}")
    x_plot = np.linspace(xlim_left, xlim_right, data.shape[0])
    y_plot = [straight_line(a, offset, x) for x in x_plot]

    # test if separator separates
    orthogonal = (-a, 1)
    # y - ax - b > 0
    # if and only if
    # (-a,1).(x,y) - b > 0
    data_1_scalar_product = np.dot(data_1, orthogonal) - offset
    data_2_scalar_product = np.dot(data_2, orthogonal) - offset
    # array of booleans
    data_1_side = data_1_scalar_product >= 0
    data_2_side = data_2_scalar_product >= 0
    # check if all points are on the same side of the separator
    # within data_1 and within data_2
    data_1_check = data_1_side == data_1_side[0]
    data_2_check = data_2_side == data_2_side[0]
    # conclusion
    # data_1 and data_2 must be on a different side
    if data_1_side[0] is not data_2_side[0]:
        # all points from data_1 must be on the same side of the
        # separator
        # same for data_2
        if np.all(data_1_check) and np.all(data_2_check):
            separation = True
        else:
            separation = False
    else:
        separation = False

    # compute the margin
    separator = np.zeros((data.shape[0], 2))
    separator[:, 0] = x_plot
    separator[:, 1] = y_plot
    distances = list()
    # compute the distance between each point in the
    # separator and the dataset
    for point in separator:
        # repeat the point to use linalg.norm
        point_repeated = np.tile(point, (data.shape[0], 1))
        # distances between this point and the dataset
        point_distances = np.linalg.norm(point_repeated - data, axis=1)
        # take the minimum
        point_distance_to_dataset = min(point_distances)
        # store this distance
        distances.append(point_distance_to_dataset)
    # the margin is the minimum of these distances
    margin = min(distances)

    # plot
    plt.plot(x_data_1, y_data_1, "o", color="sandybrown", label="orange country")
    plt.plot(x_data_2, y_data_2, "o", color="darkorchid", label="purple country")
    if separation:
        separator_color = "green"
        plt.title(f"linear separation\nmargin: {margin:.2f}")
        print(colored("linear separation", "blue", attrs=["bold"]))
        print(f"margin: {margin:.2f}")
    else:
        separator_color = "red"
        plt.title(f"no linear separation")
        print(colored("no linear separation", "yellow", attrs=["bold"]))
        print(f"margin: {margin:.2f}")
    plt.plot(x_plot, y_plot, "-", color=separator_color)
    plt.xlim(xlim_left - 60, xlim_right + 60)
    plt.ylim(ylim_bottom - 60, ylim_top + 60)
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.legend(loc="best")
    figname = f"linear_separator_a={a:.2f}_b={int(offset)}"
    plt.savefig(f"images/" + dataset + f"/{figname}.pdf")
    plt.close()


nb_test = 10
for test_index in range(nb_test):
    a = np.random.uniform(4.5, 7.5)
    b = np.random.randint(-3000, -2000)
    test_linear_separator(dataset, a, b, data_1, data_2)
