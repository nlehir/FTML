"""
    Simple data in order to illustrate the concept of
    Gaussian mixtures.

    https://scikit-learn.org/stable/modules/mixture.html

    Script adapted from
    https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng()

"""
Generate the dataset
"""
n_samples = 300

shifted_gaussian = rng.normal(size=(n_samples, 2)) + np.array([20, 20])

mat = np.array([[0.0, -0.7], [3.5, 0.7]])
stretched_gaussian_1 = rng.normal(size=(n_samples, 2)) @ mat

mat_2 = np.array([[1.0, -0.7], [0.5, 0.7]])
stretched_gaussian_2 = rng.normal(size=(n_samples, 2)) @ mat_2 + np.array([-10, 10])

X_train = np.vstack([shifted_gaussian, stretched_gaussian_1])
X_train = np.vstack([X_train, stretched_gaussian_2])

plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)
plt.savefig(os.path.join("images", "data.pdf"))
plt.close()


def learn_gmm(n_components: int) -> None:
    print(f"\nlearn gmm with {n_components} components")

    # concatenate the two datasets into the final training set

    clf = GaussianMixture(n_components=n_components, covariance_type="full")
    clf.fit(X_train)

    for index, center in enumerate(clf.means_):
        print(center)
        plt.plot(center[0], center[1], "o", label=f"center {index+1}", color="red")

    # display predicted scores by the model as a contour plot
    x = np.linspace(-20.0, 30.0)
    y = np.linspace(-20.0, 40.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    _ = plt.colorbar(CS, shrink=0.8, extend="both")
    plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)

    title = (
            "Negative log-likelihood predicted by a GMM"
            f"\nwith {n_components} components"
            f"\naverage log likelihood: {clf.score(X_train):.2f}"
            )
    plt.title(title)
    plt.legend(loc="best")
    plt.axis("tight")
    fig_name = f"gmm_{n_components}_components.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def main():
    n_components_list = [1, 2, 3, 4, 5]
    for n_components in n_components_list:
        learn_gmm(n_components=n_components)

if __name__ == "__main__":
    main()
