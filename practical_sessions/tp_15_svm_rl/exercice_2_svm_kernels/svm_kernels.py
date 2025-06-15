import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    data_path = os.path.join("data", "data.npy")
    labels_path = os.path.join("data", "labels.npy")
    data = np.load(data_path)
    labels = np.load(labels_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.33, random_state=2
    )
    y_train = y_train.ravel()
    y_test = y_test.ravel()


if __name__ == "__main__":
    main()
