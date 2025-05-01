"""
    Find the optimal number of components in order to 
    estimate the density of the digits dataset.

    We score each number of components with the Akaike information
    criterion.

    https://en.wikipedia.org/wiki/Akaike_information_criterion

    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
"""
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import os

digits = load_digits()
X_train = digits.data
y_train = digits.target


def find_nb_components() -> int:
    tab_AIC = list()
    nb_components_list = np.array(range(1, 201, 20))
    # nb_components_list = np.array(range(1, 201, 50))
    for nb_components in nb_components_list:
        print(f"{nb_components} components")
        GMM = GaussianMixture(n_components=nb_components)
        GMM.fit(X_train)
        AIC = GMM.aic(X_train)
        tab_AIC.append(AIC)
        print(f"AIC: {AIC:.2f}")

    optimal_nb_components = nb_components_list[np.argmin(tab_AIC)]

    plt.plot(nb_components_list, tab_AIC)
    fig_name = "AIC.pdf"
    fig_path = os.path.join("images", fig_name)
    title = (
        f"AIC as a function of the number of components"
        f"\noptimal value: {optimal_nb_components}"
    )
    plt.title(title)
    plt.xlabel("Number of components")
    plt.ylabel("AIC")
    plt.savefig(fig_path)
    plt.close()
    print(f"optimal nb of components found: {optimal_nb_components}")
    return optimal_nb_components


def main() -> None:
    # search for the optimal number of components
    # nb_components = find_nb_components()
    nb_components = 121

    # fit a gaussian mixture with this number of components
    covariance_type = "full"
    GMM = GaussianMixture(n_components=nb_components, covariance_type=covariance_type)
    GMM.fit(X_train)

    # plot the means of each component on a single figure
    plt.figure(figsize=[15, 12])
    n_rows = int(np.sqrt(nb_components) + 1)
    n_cols = int(np.sqrt(nb_components))
    for n in range(nb_components):
        plt.subplot(n_rows, n_cols, n + 1, xticks=[], yticks=[])
        __import__('ipdb').set_trace()
        plt.imshow(
            np.reshape(GMM.means_[n, :], newshape=(8, 8)),
            cmap="gray_r",
            vmin=0,
            vmax=16,
        )
    title = (
        "Means of the components learned by the GMM"
        f"\n{nb_components} components"
        f"\n{covariance_type} covariance"
    )
    plt.suptitle(title)
    fig_name = f"means_{nb_components}_components_{covariance_type}_covariance.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)

    # generate data according to the learned distribution
    print("Sample the GMM")
    nb_generated = 50
    data_new_X, _ = GMM.sample(nb_generated)
    n_rows = int(np.sqrt(nb_generated) + 1)
    n_cols = int(np.sqrt(nb_generated))
    plt.figure(figsize=[15, 12])
    for n in range(nb_generated):
        plt.subplot(n_rows, n_cols, n + 1, xticks=[], yticks=[])
        plt.imshow(
            np.reshape(data_new_X[n, :], newshape=(8,8)), cmap="gray_r", vmin=0, vmax=16
        )
    title = (
        "Images generated with the GMM"
        f"\n{nb_components} components"
        f"\n{covariance_type} covariance"
    )
    plt.suptitle(title)
    fig_name = f"generated_images_{nb_components}_components_{covariance_type}_covariance.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)


if __name__ == "__main__":
    main()
