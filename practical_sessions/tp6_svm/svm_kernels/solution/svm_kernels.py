import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os


def test_kernel(kernel, data, labels) -> None:
    """
    Use a kernel to attempt to linearly separate the data
    in the feature space.
    """
    print(f"\ntry kernel {kernel}")
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.33, random_state=2
    )
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # load classifier
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"accuracy: {score:.3f}")
    print(f"number of support vectors: {clf.n_support_}")

    xx, yy = np.meshgrid(np.linspace(-2, 2, 500), np.linspace(-2, 2, 500))
    # plot the decision function for each datapoint on the grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        # aspect="auto",
        origin="lower",
        cmap=plt.cm.Paired,
    )
    # contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles="dashed")
    plt.scatter(
        data[:, 0], data[:, 1], s=30, c=labels, cmap=plt.cm.Paired, edgecolors="k"
    )
    # plt.scatter(data[:, 0], data[:, 1], s=30, c=labels, edgecolors="k")
    plt.xticks(())
    plt.yticks(())
    plt.colorbar()
    plt.axis([-2, 2, -2, 2])
    title = f"SVC with {kernel} kernel\nscore (mean accuracy): {score:.2f}"
    plt.title(title)
    plt.savefig(f"decision_function_{kernel}_kernel.pdf")
    plt.close()


def main() -> None:
    # load the data
    data_path = os.path.join("data", "data.npy")
    labels_path = os.path.join("data", "labels.npy")
    data = np.load(data_path)
    labels = np.load(labels_path)

    kernels = ["rbf", "sigmoid", "poly", "linear"]
    for kernel in kernels:
        test_kernel(kernel, data, labels)


if __name__ == "__main__":
    main()
