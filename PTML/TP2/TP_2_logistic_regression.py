import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = np.load("./data/logistic_regression/data.npy")
labels = np.load("./data/logistic_regression/labels.npy")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)
d = data.shape[1]

n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"n train: {n_train}")
print(f"n test: {n_test}")
