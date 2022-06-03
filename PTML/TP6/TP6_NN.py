from sklearn import datasets
import os
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
class_1 = np.where(labels == 1)[0]
class_2 = np.where(labels == 0)[0]
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
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.33, random_state=2
)
n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"n train: {n_train}")
print(f"n test: {n_test}")
class_1_train = np.where(y_train == 1)[0]
class_2_train = np.where(y_train == 0)[0]
class_1_test = np.where(y_test == 1)[0]
class_2_test = np.where(y_test == 0)[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def forward_pass(x, wh, theta):
    """
    wh: matrix of size (m, d+1) containing weights between
    the input layer and the hidden layer.

    theta: vector of size (m+1, 1) containing the weights
    between the hidden layer and the output.

    check the dimensions of the arrays !
    """
    pre_h = wh @ np.append(x, 1)
    h = sigmoid(pre_h)
    pre_y = theta @ np.append(h, 1)
    y_hat = sigmoid(pre_y)
    return pre_h, h, pre_y, y_hat


def gradients(x, y, pre_h, h, pre_y, y_hat):
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
    dy_hat_dpre_y = sigmoid_derivative(pre_y)
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
    dh_dpre_h = sigmoid_derivative(pre_h)
    # vector with m components
    # this is an elementwise product
    dl_dpre_h = dl_dh * dh_dpre_h
    dl_dwh = np.matrix(dl_dpre_h).T * np.matrix(np.append(x, 1))
    return dl_dtheta, dl_dwh


gammas_list = [0.01, 0.1, 1, 10]
# gammas_list = [1]
m_list = [5, 10, 20, 30]
m_list = [2, 5, 10]
nb_iterations = int(1e3)
d = len(data[0])

"""
    initialization
"""
for m in m_list:
    if not os.path.exists(f"./images_nn/m={m}"):
        os.makedirs(f"./images_nn/m={m}")
    for gamma in gammas_list:
        print(f"---\nm: {m}")
        print(f"gamma: {gamma}")
        wh = np.random.rand(m, d + 1)
        theta = np.random.rand(m + 1)

        """
            algorithm
        """
        times = list()
        train_errors = list()
        test_errors = list()

        for t in range(nb_iterations):
            i = np.random.randint(n_train)
            x, y = X_train[i], y_train[i]
            pre_h, h, pre_y, y_hat = forward_pass(x, wh, theta)
            dl_dtheta, dl_dwh = gradients(x, y, pre_h, h, pre_y, y_hat)

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
                    train_error += (y_hat - y) ** 2 / 2
                train_error /= n_train
                train_errors.append(train_error)
                # test error
                test_error = 0
                for k in range(n_test):
                    x, y = X_test[k], y_test[k]
                    pre_h, h, pre_y, y_hat = forward_pass(x, wh, theta)
                    test_error += (y_hat - y) ** 2 / 2
                test_error /= n_test
                test_errors.append(test_error)

        plt.plot(np.log10(times), train_errors, label="train eror")
        plt.plot(np.log10(times), test_errors, label="test eror")
        plt.legend(loc="best")
        plt.xlabel("log10 iteration")
        plt.ylabel("mean squared error")
        plt.ylim([-0.01, 0.3])
        plt.yticks([0, 0.15, 0.3])
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
        plt.savefig(f"./images_nn/m={m}/" + edited_figname + ".pdf")
        plt.close()

        """
            predict y_hat for one input
        """

        def predict(x):
            pre_h, h, pre_y, y_hat = forward_pass(x, wh, theta)
            return y_hat

        """
            Show separation
        """
        ngrid = 100
        matrix_predictions = np.zeros((ngrid, ngrid))
        xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, ngrid), np.linspace(-1, 1.5, ngrid))
        uu = np.c_[xx.ravel(), yy.ravel()]
        Z = np.zeros(len(uu))
        for i in range(uu.shape[0]):
            Z[i] = predict(uu[i])
        Z = Z.reshape(xx.shape)
        plot = plt.imshow(
            Z,
            vmin=0,
            vmax=1,
            alpha=0.5,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            origin="lower",
            cmap="GnBu",
        )

        plt.scatter(
            X_train[class_1_train, 0],
            X_train[class_1_train, 1],
            label="class_1 train: y=1",
            edgecolors="k",
        )
        plt.scatter(
            X_train[class_2_train, 0],
            X_train[class_2_train, 1],
            label="class_2 train: y=0",
            edgecolors="k",
        )
        plt.scatter(
            X_test[class_1_test, 0],
            X_test[class_1_test, 1],
            label="class_1 test: y=1",
            edgecolors="k",
        )
        plt.scatter(
            X_test[class_2_test, 0],
            X_test[class_2_test, 1],
            label="class_2 test: y=0",
            edgecolors="k",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(plot, shrink=0.6, ticks=[0, 0.5, 1])
        title = (
            f"prediction function after SGD\n"
            + f"number of iterations: {nb_iterations:.2E}\n"
            + f"m={m}\n"
            + r"$\gamma=$"
            + f"{gamma:.2f}"
        )
        plt.title(title)
        plt.legend(loc="best", prop={"size": 6})
        plt.tight_layout()
        figname = f"prediction_it_{nb_iterations:.2E}_m_{m}_gam_{gamma}"
        # remove dots in figname
        edited_figname = figname.replace(".", "_")
        plt.savefig(f"./images_nn/m={m}/" + edited_figname + ".pdf")
        plt.close()
