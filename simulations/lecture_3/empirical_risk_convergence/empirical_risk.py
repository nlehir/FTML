import numpy as np
import matplotlib.pyplot as plt


def f1(X: np.ndarray) -> np.ndarray:
    y_pred = np.zeros(len(X))
    y_pred[X == 1] = 1
    y_pred[X == 0] = 0
    return y_pred


def f2(X: np.ndarray) -> np.ndarray:
    y_pred = np.zeros(len(X))
    y_pred[X == 1] = 0
    y_pred[X == 0] = 1
    return y_pred


def f3(X: np.ndarray) -> np.ndarray:
    y_pred = np.ones(len(X))
    return y_pred


def sample_dataset(n_samples, p, q):
    """
    Sample a dataset of n samples according to
    the joint law.

    X ~ B(1/2)
    Y ~ B(p) if X=1
    Y ~ B(q) if X=0
    """
    rng = np.random.default_rng()

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
    bernoulli[bernoulli == 1] = p
    bernoulli[bernoulli == 0] = q

    # generate output data
    y = rng.binomial(1, bernoulli)

    return X, y


def compute_empirical_risk(f, X, y):
    """
    Compute empirical risk of predictor
    on the dataset

    Parameters:
        X: 1D array
        y: 1D array
        f: predictor

    Returns:
        empirical risk

    We use the "0-1"-loss
    """
    n_samples = X.shape[0]
    y_pred = f(X)
    empirical_risk = len(np.where(y - y_pred)[0]) / n_samples
    return empirical_risk


def plot_convergence(X, y,generalization_error_f, f):
    name = f.__name__
    max_n_samples = len(X)

    # Monte carlo
    empirical_risks = list()
    for n in range(1, max_n_samples):
        X_sample = X[:n]
        y_sample = y[:n]
        empirical_risks.append(compute_empirical_risk(f, X_sample, y_sample))

    # plot results
    plt.plot(
        range(1, max_n_samples),
        empirical_risks,
        "o",
        markersize=2,
        alpha=0.3,
        label=r"$R_n(f)$" + " empirical risk",
    )
    plt.plot(
        range(1, max_n_samples),
        (max_n_samples - 1) * [generalization_error_f],
        color="hotpink",
        label="real risk / generalization error",
    )

    # finish plot
    plt.xlabel("number of samples")
    plt.legend(loc="best")
    plt.title(
        f"{name}\n"
        "Convergence of the empirical risk to the real risk"
        + "\n"
        + f"R({name})"
        + f"={generalization_error_f:.2f}"
    )
    figname = f"empirical_risk_and_generalization_error_{name}.pdf"
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def main():
    # constants
    p = 1 / 3
    q = 3 / 4

    # sample the dataset
    max_n_samples = int(3e3)
    X, y = sample_dataset(max_n_samples, p, q)

    generalization_error_f_1 = (1 - p) / 2 + q / 2
    generalization_error_f_2 = 1 - generalization_error_f_1
    generalization_error_f_3 = (1 - p) / 2 + (1 - q) / 2

    plot_convergence(X, y, generalization_error_f_1, f1)
    plot_convergence(X, y, generalization_error_f_2, f2)
    plot_convergence(X, y, generalization_error_f_3, f3)


if __name__ == "__main__":
    main()
