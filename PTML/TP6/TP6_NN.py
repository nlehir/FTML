from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

"""
    load and plot data
"""
n = 300
np.random.seed(0)
data, labels = datasets.make_moons(n, noise=0.1)
labels = labels.reshape(n)
class_1 = np.where(labels==1)[0]
class_2 = np.where(labels==0)[0]
plt.figure(figsize=(10, 7))
plt.scatter(data[class_1, 0], data[class_1, 1], label="class_1: y=1")
plt.scatter(data[class_2, 0], data[class_2, 1], label="class_2: y=0")
plt.xlabel("x")
plt.ylabel("y")
plt.title("data to classify")
plt.legend(loc="best")
plt.savefig("./images_nn/data_nn.pdf")
plt.close()

"""
    split the data into a training set and a test set
"""
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=2)
n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"n train: {n_train}")
print(f"n test: {n_test}")
class_1_train = np.where(y_train==1)[0]
class_2_train = np.where(y_train==0)[0]
class_1_test = np.where(y_test==1)[0]
class_2_test = np.where(y_test==0)[0]


"""
    implement sigmoids
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


"""
    forward pass
"""
def forward_pass(x, wh, theta):
    return 1


"""
    compute gradients
"""
def gradients(x, y, zh, ah, zo, ao):
    return 1


"""
    initialization
"""


"""
    algorithm
"""
