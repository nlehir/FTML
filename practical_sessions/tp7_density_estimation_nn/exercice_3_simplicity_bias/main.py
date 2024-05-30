"""
    Approximate a dataset with SGD on a one hidden layer neural network
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from utils_solution import compute_gradients, forward_pass

data_folder = "data"
inputs = np.load(os.path.join(data_folder, "inputs.npy"))
outputs = np.load(os.path.join(data_folder, "outputs.npy"))
bayes_predictions = np.load(os.path.join(data_folder, "bayes_predictions.npy"))

GAMMAS_LIST = [0.01, 0.1]
M_LIST = [2, 5, 100]
NB_ITERATIONS = int(1e5)

# split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    inputs, outputs, test_size=0.10, random_state=2
)
n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"n train: {n_train}")
print(f"n test: {n_test}")


def learn_neural_network(m: int, gamma: float):
    """
    Learn a neural network with one hidden layer of m (actually m+1) neurons in order
    to fit the data, by stochastic gradient descent.

    m (int): number of neurons in the hidden layer
    gamma (float): learning rate

    EDIT THE INITIALIZATIONS !
    You can also experiment with different initializations of the parameters.
    """

    # Initialize the weights
    # We use Glorot initialization
    phi = np.random.uniform(-math.pi, math.pi, (1, m))
    wh = 1 / math.sqrt(m) * np.vstack((np.cos(phi), np.sin(phi)))
    __import__('ipdb').set_trace()
    theta = np.random.uniform(-1 / math.sqrt(m), 1 / math.sqrt(m), m + 1)

    """
    Perform SGD on the empirical risk minimization problem.
    """
    times = list()
    train_errors = list()
    test_errors = list()
    for t in range(NB_ITERATIONS):
        # sample a random index
        i = np.random.randint(n_train)
        x, y = X_train[i], y_train[i]

        # compute forward pass
        outputs = forward_pass(X=x, wh=wh, theta=theta)
        pre_h = outputs["pre_h"]
        h = outputs["h"]
        pre_y = outputs["pre_y"]
        y_hat = outputs["y_hat"]

        # compute gradients
        gradients = compute_gradients(x, y, pre_h, h, pre_y, y_hat, theta)
        dl_dtheta = gradients["dl_dtheta"]
        dl_dwh = gradients["dl_dwh"]

        # update the weights
        wh -= gamma * dl_dwh
        theta -= gamma * dl_dtheta

        # Store scoring only for some steps in order to save computation time.
        if (t - 1) % int(NB_ITERATIONS / 50) == 0:
            print(f"{t}/{NB_ITERATIONS}")

            preds_train = forward_pass(X=X_train, wh=wh, theta=theta)["y_hat"]
            train_loss = (np.linalg.norm(y_train - preds_train) ** 2) / (2 * n_train)
            preds_test = forward_pass(X=X_test, wh=wh, theta=theta)["y_hat"]
            test_loss = (np.linalg.norm(y_test - preds_test) ** 2) / (2 * n_test)

            times.append(t)
            train_errors.append(train_loss)
            test_errors.append(test_loss)

    """
    Plot the results
    """
    plt.plot(times, train_errors, label="train eror")
    plt.plot(times, test_errors, label="test eror")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="best")
    plt.xlabel("log10 iteration")
    plt.ylabel("mean squared error")
    title = (
        f"train and test learning curves\n" + f"m={m}\n" + r"$\gamma=$" + f"{gamma:.2f}"
    )
    plt.title(title)
    plt.tight_layout()
    figname = f"learning_curves_it_{NB_ITERATIONS:.2E}_m_{m}_gam_{gamma}"
    figname = f"{figname.replace('.', '_')}.pdf"
    plt.savefig(os.path.join("images", figname))
    plt.close()

    predictions = forward_pass(X=inputs, wh=wh, theta=theta)["y_hat"]
    plt.plot(X_train, y_train, "o", label="train", alpha=0.8)
    plt.plot(X_test, y_test, "o", label="test", alpha=0.8)
    plt.plot(inputs, bayes_predictions, label="bayes predictor", color="aqua")
    plt.plot(inputs, predictions, label="neural network")
    plt.xlabel("input")
    plt.ylabel("output")
    title = f"neural net prediction\nm={m}\n" + r"$\gamma=$" + f"{gamma:.2f}"
    plt.title(title)
    plt.tight_layout()
    figname = f"prediction_it_{NB_ITERATIONS:.2E}_m_{m}_gam_{gamma}"
    figname = f"{figname.replace('.', '_')}.pdf"
    plt.legend(loc="best")
    plt.savefig(os.path.join("images", figname))
    plt.close()


def main() -> None:
    for m in M_LIST:
        for gamma in GAMMAS_LIST:
            print(f"---\nm: {m}")
            print(f"gamma: {gamma}")
            learn_neural_network(m=m, gamma=gamma)


if __name__ == "__main__":
    main()
