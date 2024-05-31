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
    NB_Y = 100


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

    # compute the probability density for each sample in XY
    # we apply an exponential because the score_samples
    # returned are logs of the density probabilities.
    p_x_y = np.exp(gmm.score_samples(XY))
    # reorganize the densities in order to sum them
    # more conveniently afterwards
    # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # We need to sum over y values.
    p_y_x = np.reshape(p_x_y, newshape=(NB_Y, nb_x))

    p_x = np.sum(p_y_x, axis=0)

    # vectorized approximatations of the conditional expected values of y for
    # each value in x.
    p_y_sachant_x = p_y_x / p_x
    preds = np.sum(Y * p_y_sachant_x, axis=0)
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

    # compute the optimal number of components according to the Bayesian
    # information criterion
    BIC_list = list()
    nb_components_list = range(1, 11)
    for nb_components in nb_components_list:
        gmm = GaussianMixture(n_components=nb_components, covariance_type="full")
        gmm.fit(data)
        bic = gmm.bic(data)
        BIC_list.append(bic)
        print(f"{nb_components} components: BIC={bic:.2f}")

    # fit a gmm with this optimal number of components
    gmm = GaussianMixture(n_components=2, covariance_type="full")
    gmm.fit(data)

    plt.figure()
    plt.plot(range(1, 11), BIC_list)
    plt.xlabel("number of components")
    plt.ylabel("BIC")
    plt.title("BIC as a function of the number of components")
    fig_name = "bic.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)

    nb_x = 50
    x_min = data[:, 0].min()
    x_max = data[:, 0].max()
    x = np.linspace(x_min, x_max, nb_x)
    preds = prediction_gmm(gmm=gmm, x=x)

    plt.figure(figsize=[12, 10])
    plt.scatter(data[:, 0], data[:, 1], label="data")
    plt.plot(x, preds, color="r", label="prediction")
    plt.title("Old Faithful geyser dataset")
    plt.legend(loc="best")
    plt.xlabel("eruption duration (min)")
    plt.ylabel("time before the next eruption (min)")
    plt.title(f"Old Faithful eruption dataset\n{NB_Y} steps in y discretization")
    fig_name = f"preds_old_faithful_prediction_nb_y_{NB_Y}.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)


if __name__ == "__main__":
    main()
