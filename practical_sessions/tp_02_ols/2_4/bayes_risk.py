import numpy as np
import matplotlib.pyplot as plt

d = 10
sigma = 0.2


def main():

    n_list = np.linspace(10, 100000, 100)
    test_errors = list()
    for n in n_list:
        test_errors.append(test_error(n_samples=int(n)))

    y_bayes_risk = np.ones_like(n_list) * sigma**2
    label = r"$\sigma^2$" f" (Bayes risk)"
    plt.ylabel("Test error")
    plt.xlabel("Number of test samples")
    plt.plot(n_list, y_bayes_risk, label=label)
    plt.plot(
        n_list,
        test_errors,
        "o",
        label="test error",
        alpha=0.4,
        color="aqua",
    )
    plt.title(
        "Test error of the Bayes estimator as a function of the number of test samples",
        fontsize=10,
    )
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig(f"test_error_d_{d}.pdf")
    plt.close()


def test_error(n_samples):
    rng = np.random.default_rng()
    theta_star = rng.uniform(low=0, high=1, size=(d, 1))
    X = rng.uniform(low=-1, high=2, size=(n_samples, d))
    epsilon = rng.normal(loc=0, scale=sigma, size=(n_samples, 1))
    y = X @ theta_star + epsilon
    y_pred_bayes = X @ theta_star
    test_error = (np.linalg.norm(y - y_pred_bayes) ** 2) / n_samples
    return test_error


if __name__ == "__main__":
    main()
