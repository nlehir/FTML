import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

X_train = np.load(os.path.join("data", "X_train.npy"))
X_test = np.load(os.path.join("data", "X_test.npy"))
y_train = np.load(os.path.join("data", "y_train.npy"))
y_test = np.load(os.path.join("data", "y_test.npy"))


def main():
    pca = PCA()
    projected_X_train = pca.fit_transform(X_train)

    # proj data 2D
    plt.scatter(
        projected_X_train[:, 0],
        projected_X_train[:, 1],
        c=y_train,
        alpha=0.9,
    )
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.colorbar()
    title = "projection on the 2 principal components"
    plt.title(title)
    plt.savefig("projected_X_train_2D.pdf")
    plt.close("all")

    # proj X_train 3D
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plot  = ax.scatter3D(
        projected_X_train[:, 0],
        projected_X_train[:, 1],
        projected_X_train[:, 2],
        c = y_train,
        alpha=0.9
    )
    fig.colorbar(plot)
    title = "projection on the 3 principal components"
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.title(title)
    plt.savefig("projected_X_train_3D.pdf")
    plt.close()

    # logistic regression with projection
    X_train_proj = pca.fit_transform(X_train)
    X_test_proj = pca.fit_transform(X_test)
    test_proj_dimension(X_train_proj, X_test_proj, y_train, y_test, 2)
    test_proj_dimension(X_train_proj, X_test_proj, y_train, y_test, 3)

    # logistic regression without projection
    clf = LogisticRegression().fit(X_train, y_train)
    print("\ntest logistic regression without projection")
    print(f"train accuracy= {clf.score(X_train, y_train):.2F}")
    print(f"test accuracy= {clf.score(X_test, y_test):.2F}")


def test_proj_dimension(X_train_proj, X_test_proj, y_train, y_test, dimension):
    print(f"\ntest projection in {dimension} dimensions")

    X_train_proj_trunc = X_train_proj[:, :dimension]
    X_test_proj_trunc = X_test_proj[:, :dimension]

    clf = LogisticRegression().fit(X_train_proj_trunc, y_train)

    print(f"train accuracy= {clf.score(X_train_proj_trunc, y_train):.2F}")
    print(f"test accuracy= {clf.score(X_test_proj_trunc, y_test):.2F}")


if __name__ == "__main__":
    main()
