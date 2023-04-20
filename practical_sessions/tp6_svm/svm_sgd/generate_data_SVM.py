import os

import matplotlib.pyplot as plt
import numpy as np


def generate_data(
    mean_1: tuple, std_1: float, mean_2: tuple, std_2: float, n_cluster: int, name: str
):
    """
    Generate a dataset used to study the linear separation problem.

    We generate two "subdatasets" and concatenate them.

    :param mean_1: mean of the first subdataset.
    :param std_1: standard deviation of the first dataset
    :param mean_2: mean of the second subdataset.
    :param std_2: standard deviation of the second dataset
    :param n_cluster: number of points in a subdataset
    :param name: name of the datset
    :type mean_1: tuple of floats
    :type mean_2: tuple of floats
    :type std_1: float
    :type std_2: float
    """
    # generate data
    data_1 = np.random.normal(loc=mean_1, scale=std_1, size=(n_cluster, 2))
    data_2 = np.random.normal(loc=mean_2, scale=std_2, size=(n_cluster, 2))
    data = np.concatenate((data_1, data_2))
    class_1 = np.zeros(data_1.shape[0])
    class_2 = np.ones(data_2.shape[0])
    labels = np.concatenate((class_1, class_2))

    # save
    if not os.path.exists(os.path.join("data/", name)):
        os.makedirs(os.path.join("data/", name))
    np.save("data/" + name + "/data", data)
    np.save("data/" + name + "/labels", labels)

    # plot
    plt.plot(data_1[:, 0], data_1[:, 1], "o", color="sandybrown", label="class 1")
    plt.plot(data_2[:, 0], data_2[:, 1], "o", color="darkorchid", label="class 2")
    plt.legend(loc="best")
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.title("linear classification problem")
    plt.savefig("data_to_separate_" + name + ".pdf")
    plt.close()


# first dataset
mean_1 = (600, -10)
std_1 = 40
mean_2 = (200, 130)
std_2 = 50
n_cluster = 200
generate_data(mean_1, std_1, mean_2, std_2, n_cluster, "dataset_1")

# second dataset
mean_1 = (600, 2)
std_1 = 70
mean_2 = (180, 130)
std_2 = 80
generate_data(mean_1, std_1, mean_2, std_2, n_cluster, "dataset_2")

# third dataset
mean_1 = (600, -10)
std_1 = 150
mean_2 = (200, 130)
std_2 = 150
generate_data(mean_1, std_1, mean_2, std_2, n_cluster, "dataset_3")
