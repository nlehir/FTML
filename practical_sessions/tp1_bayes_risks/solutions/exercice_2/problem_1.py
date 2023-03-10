"""
Study the Bayes estimator for problem 1.
We will prove the results used here during the lectures.
"""

import numpy as np


def main() -> None:
    """
    Simulate problem 1
    """

    # instantiate a Pseudo-random number generator (PRNG)
    rng = np.random.default_rng()

    # global parameters
    n_samples = int(1e6)

    # generate input data
    # they are uniformly distributed
    # 0s and 1s
    # If we do not cast to floats, we will have a mistake
    # in the next step (all Bernoulli parameters would be
    # set to 0)
    X = rng.integers(0, 2, n_samples).astype(float)

    # for each input, get the parameter of the
    # corresponding Bernoulli law.
    bernoulli = X.copy()
    bernoulli[bernoulli == 1] = 0.6
    bernoulli[bernoulli == 0] = 0.4

    # generate output data
    y = rng.binomial(1, bernoulli)

    # generate predictions with the Bayes estimator
    # When doing classification with the "0-1" loss,
    # the Bayes estimator predict the most probable output
    # for each input. (we will show this during the class)
    # In that case, it turns out that it corresponds exactly to
    # predicting X, but note that this will not always be the case.
    y_pred_bayes = X

    # generate predictions for a bad etimator (uniformly random)
    y_pred_bad = rng.integers(0, 2, n_samples).astype(float)

    # compute the empirical risk for the Bayes estimator
    empirical_risk_bayes = len(np.where(y - y_pred_bayes)[0]) / n_samples

    # empirical risk bad estimator
    empirical_risk_bad_estimator = len(np.where(y - y_pred_bad)[0]) / n_samples

    print("\nX")
    print(X)
    print("\ny")
    print(y)
    print("\ny pred squared loss")
    print(y_pred_bayes.astype(int))
    print("\nempirical risk for bayes predictor squared loss")
    print(empirical_risk_bayes)
    print("\nempirical risk for bad predictor squared loss")
    print(empirical_risk_bad_estimator)


if __name__ == "__main__":
    main()
