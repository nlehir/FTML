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
    """
    EDIT THIS FUNCTION

    Find the optimal number of components according
    to the Akaike information criterion (AIC).
    """
    optimal_nb_components = 2
    return optimal_nb_components


def main() -> None:
    # search for the optimal number of components
    nb_components = find_nb_components()
    # nb_components = 121

    # fit a gaussian mixture with this number of components
    covariance_type = "full"
    GMM = GaussianMixture(n_components=nb_components, covariance_type=covariance_type)
    GMM.fit(X_train)

    # plot the means of each component on a single figure

    # generate data according to the learned distribution


if __name__ == "__main__":
    main()
