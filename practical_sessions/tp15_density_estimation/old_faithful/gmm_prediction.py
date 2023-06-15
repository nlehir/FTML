"""
    Use the Gaussian mixtures as a prediction tool
    for the Old Faithful geyser dataset.

    https://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat
"""
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

data = np.loadtxt("old_faithful.txt")
# remove the first column, that only contains indices
data = data[:, 1:3]

# number of samples to approximate the integrals in the expected value
# computation
if len(sys.argv) > 1:
    NB_Y = int(sys.argv[1])
else:
    NB_Y = 10


def prediction_gmm(gmm: GaussianMixture, x: np.ndarray) -> np.ndarray:
    """
    Predict the conditional expectation of the
    time between eruptions (y) as a function of the eruption duration (x).

    We approximate the integrals involved in the computation of the
    expected value as finite sums, thanks to Riemann sums.

    https://en.wikipedia.org/wiki/Riemann_sum

    However, since there will be an integral at the numerator
    AND at the denominator, and as the integrals are computed over the
    same range of y values, we do not need to multiply by the classical
    (b-a)/n, like when we approximate an integral between two real numbers
    a and b with n samples.

    We also preferrably perform this computation in a vectorized way.
    """
    nb_x = len(x)
    y_min = 10
    y_max = 130
    y = np.linspace(y_min, y_max, num=NB_Y)

    # create an array that contains all (x,y) samples
    # from the list of x values and the list of y values.
    X, Y = np.meshgrid(x, y)
    # XY will be of shape (nb_x*NB_Y, 2)
    XY = np.array([X.ravel(), Y.ravel()]).T

    """
    EDIT FROM HERE
    """

    # compute the probability density for each sample in XY
    # we apply an exponential because the score_samples
    # returned are logs of the density probabilities.
    p_x_y = 1
    # reorganize the densities in order to sum them
    # more conveniently afterwards
    # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # We need to sum over y values.
    p_x_y = 1

    p_x = 1

    # vectorized approximatations of the conditional expected values of y for
    # each value in x.
    preds = 1
    return preds


def main() -> None:
    # first plot the data
    plt.figure(figsize=[12, 10])
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("eruption duration (min)")
    plt.ylabel("time before the next eruption (min)")
    plt.title("Old Faithful eruption dataset")
    fig_name = "data_old_faithful.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)


    """
    EDIT FROM HERE
    """
    # compute the optimal number of components according to the Bayesian
    # information criterion

    # fit a gmm with this optimal number of components

    # compute the predictions for a sequence of x values

if __name__ == "__main__":
    main()
