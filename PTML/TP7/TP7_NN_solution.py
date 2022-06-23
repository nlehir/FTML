import os
import math
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from TP7_utils import forward_pass, gradients

inputs = np.load("data/inputs.npy")
outputs = np.load("data/outputs.npy")
targets = np.load("data/targets.npy")

"""
    split the data into a training set and a test set
"""
X_train, X_test, y_train, y_test = train_test_split(
    inputs, outputs, test_size=0.10, random_state=2
)
n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"n train: {n_train}")
print(f"n test: {n_test}")


gammas_list = [0.01, 0.1, 1, 10]
gammas_list = [0.05, 0.01]
gammas_list = [0.01]
m_list = [5, 10, 20, 30, 40, 50, 100]
m_list = [30, 40, 50, 60, 100]
m_list = [30]
nb_iterations = int(1e5)

"""
    initialization
"""
for m in m_list:
    for gamma in gammas_list:
        print(f"---\nm: {m}")
        print(f"gamma: {gamma}")
        """
            Glorot initialization
        """
        phi = np.random.uniform(-math.pi, math.pi, (m, 1))
        wh = 1 / math.sqrt(m) * np.column_stack((np.cos(phi), np.sin(phi)))
        theta = np.random.uniform(-1 / math.sqrt(m), 1 / math.sqrt(m), m + 1)
        times = list()
        train_errors = list()
        test_errors = list()

        for t in range(nb_iterations):
            i = np.random.randint(n_train)
            x, y = X_train[i], y_train[i]
            pre_h, h, pre_y, y_hat = forward_pass(x, wh, theta)
            dl_dtheta, dl_dwh = gradients(x, y, pre_h, h, pre_y, y_hat, theta)

            wh -= gamma * dl_dwh
            theta -= gamma * dl_dtheta

            # store errors only for some steps
            if (t - 1) % int(nb_iterations / 50) == 0:
                print(f"{t}/{nb_iterations}")
                # store time
                times.append(t)
                # train error
                train_error = 0
                for j in range(n_train):
                    x, y = X_train[j], y_train[j]
                    pre_h, h, pre_y, y_hat = forward_pass(x, wh, theta)
                    # train_error += abs(y_hat - y) / 2
                    train_error += (y_hat - y) ** 2 / 2
                train_error /= n_train
                train_errors.append(train_error)
                # test error
                test_error = 0
                for k in range(n_test):
                    x, y = X_test[k], y_test[k]
                    pre_h, h, pre_y, y_hat = forward_pass(x, wh, theta)
                    # test_error += abs(y_hat - y) / 2
                    test_error += (y_hat - y) ** 2 / 2
                test_error /= n_test
                test_errors.append(test_error)
        plt.plot(times, train_errors, label="train eror")
        plt.plot(times, test_errors, label="test eror")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(loc="best")
        plt.xlabel("log10 iteration")
        plt.ylabel("mean squared error")
        title = (
            f"train and test learning curves\n"
            + f"m={m}\n"
            + r"$\gamma=$"
            + f"{gamma:.2f}"
        )
        plt.title(title)
        plt.tight_layout()
        figname = f"learning_curves_it_{nb_iterations:.2E}_m_{m}_gam_{gamma}"
        # remove dots in figname
        edited_figname = figname.replace(".", "_")
        plt.savefig(f"./images_nn/" + edited_figname + ".pdf")
        plt.close()

        """
            predict y_hat for one input
        """

        def predict(x):
            pre_h, h, pre_y, y_hat = forward_pass(x, wh, theta)
            return y_hat

        predictions = [predict(x) for x in inputs]
        plt.plot(X_train, y_train, "o", label="train", alpha=0.8)
        plt.plot(X_test, y_test, "o", label="test", alpha=0.8)
        plt.plot(inputs, targets, label="target", color="aqua")
        plt.plot(inputs, predictions, label="predictions")
        plt.xlabel("input")
        plt.ylabel("output")
        title = f"neural net prediction\n" + f"m={m}\n" + r"$\gamma=$" + f"{gamma:.2f}"
        plt.title(title)
        plt.tight_layout()
        figname = f"prediction_it_{nb_iterations:.2E}_m_{m}_gam_{gamma}"
        # remove dots in figname
        edited_figname = figname.replace(".", "_")
        plt.legend(loc="best")
        plt.savefig(f"./images_nn/" + edited_figname + ".pdf")
        plt.close()
