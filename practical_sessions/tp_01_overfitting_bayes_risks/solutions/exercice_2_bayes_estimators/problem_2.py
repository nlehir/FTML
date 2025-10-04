"""
Study the Bayes estimator for problem 2.
We will prove the results used here during the lectures.
"""

import numpy as np
from scipy.stats import binom


def bayes_estimator(n_binom_streams: np.ndarray, p_streams: float) -> np.ndarray:
    """
    In this case (using the squared loss), the Bayes estimator
    predicts the conditional expected value of Y, given that the input
    equals some value x.

    Here, we can hence use the expected value of the binomial law, and
    compute it directly for all inputs in a numpy array.
    """
    return n_binom_streams * p_streams


def compute_bayes_risk(n_x: float, p_x: float, p_all) -> np.float64:
    """
    Compute the analytical value of the Bayes risk.

    By definition, the Bayes risk is an expected value. So here, we must
    compute the expected value of a discrete random variable.
    But we can compute it exactly because we have access to all
    necessary probabilities (which will not be the case
    in most practical applications).

    Remember that the value of the Bayes risk will always depend on the
    loss function used.
    """
    # probas for each individual value of x
    # which follows a binomial law
    probas = binom.pmf(np.arange(n_x + 1), n_x, p_x)

    # conditional variance of the target given fixed x
    # see class to understand why we compute this
    n_binom_all = 3 ** (1 + np.arange(n_x + 1))
    variances = n_binom_all * p_all * (1 - p_all)

    # weighted sum to get the expected value, which
    # is the Bayes risk
    bayes_risk = (probas * variances).sum()

    return bayes_risk


def bad_estimator(X) -> np.ndarray:
    """
    Compare with a bad etimator
    """
    return 1000 * X


def main() -> None:
    """
    Simulate problem 2
    """
    # instantiate a Pseudo-random number generator (PRNG)
    rng = np.random.default_rng()

    # global parameters
    n_samples = int(1e6)

    # parameters for X
    n_x = 20
    p_x = 0.2

    # parameters for y
    # p parameter of the binomial law of Y, given x
    p_streams = 0.5

    # generate input data
    X = 1 + rng.binomial(n_x, p_x, size=(n_samples,))

    # generate output data
    y = rng.binomial(3**X, p_streams, size=(n_samples,))

    # generate predictions bayes
    y_pred_bayes = bayes_estimator(3**X, p_streams)

    # compute bayes risk
    # for the squared loss
    bayes_risk = compute_bayes_risk(n_x, p_x, p_streams)

    # compute empirical risk bayes
    empirical_risk_bayes_estimator = ((y_pred_bayes - y) ** 2).mean()

    # predictions bad
    y_pred_bad_estimator = bad_estimator(X)
    empirical_risk_bad = ((y_pred_bad_estimator - y) ** 2).mean()

    print("\nX")
    print(X)
    print("\ny")
    print(y)
    print("\ny pred bayes")
    print(y_pred_bayes.astype(int))
    print("\ny pred bad")
    print(y_pred_bad_estimator.astype(int))
    print("\nBayes risk squared loss")
    print(bayes_risk)
    print("\nempirical risk for bayes predictor squared loss")
    print(empirical_risk_bayes_estimator)
    print("\nempirical risk for bad predictor squared loss")
    print(empirical_risk_bad)


if __name__ == "__main__":
    main()
