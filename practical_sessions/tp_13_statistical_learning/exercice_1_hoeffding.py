import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.01
N_REPEATS = 1000
N_MIN = 10
N_MAX = 10000
EXPECTED_VALUE = 0.5
rng = np.random.default_rng()

def clean_filename(name: str):
    name = name.replace(".", "_")
    name = name.replace(" ", "_")
    return name

def main():
    n_values = np.arange(start=N_MIN, stop=N_MAX)
    n_values_probas = np.arange(start=N_MIN, stop=N_MAX, step=N_MAX//100)
    probas = list()
    print(f"{EPSILON=}")
    for n in n_values_probas:
        # if n % (N_MAX//10) == 0:
        #     print(f"{n=}")
        print(f"{n=}")
        probas.append(estimate_proba(n))

    hoeffding = 2 * np.exp(-(2*n_values*EPSILON**2))
    plt.plot(n_values_probas, probas, label=r"$P(|\bar{\mu}-\mu|\geq \epsilon)$", marker="o", alpha=0.7)
    plt.plot(n_values, hoeffding, label=r"$2\exp(-\frac{2n\epsilon^2}{(b-a)^2})$")
    title = (
            f"Hoeffding's inequality\n"
            f"Bernoulli 1/2\n"
            r"$\epsilon=$"
            f"{EPSILON}"
            )
    plt.title(title)
    plt.xlabel("n")
    plt.legend(loc="best")
    fig_name = f"Hoeffding_epsilon_{EPSILON:.3f}"
    fig_name = clean_filename(fig_name)
    plt.tight_layout()
    plt.savefig(f"{fig_name}.pdf")
    plt.close()

def estimate_proba(n):
    samples = rng.integers(low=0, high=2, size=(N_REPEATS, n))
    averages = np.mean(samples, axis=1)
    abs_differences = np.abs(averages - EXPECTED_VALUE)
    diff_outside = abs_differences >= EPSILON
    rate = diff_outside.sum() / len(samples)
    return rate



if __name__ == "__main__":
    main()
