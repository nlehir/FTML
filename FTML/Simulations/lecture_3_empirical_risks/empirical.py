import numpy as np
import matplotlib.pyplot as plt


def f1(x):
    if x == 1:
        return 1
    elif x == 0:
        return 0
    else:
        raise ValueError("incorrect input")


def f2(x):
    if x == 1:
        return 0
    elif x == 0:
        return 1
    else:
        raise ValueError("incorrect input")


def f3(x):
    if x == 1:
        return 1
    elif x == 0:
        return 1
    else:
        raise ValueError("incorrect input")


def sample_dataset(n, p, q):
    """
        Sample a dataset of n samples according to
        the joint law.

        X ~ B(1/2)
        Y ~ B(p) if X=1
        Y ~ B(q) if X=0
    """
    X = np.random.randint(0, 2, n)
    Y = np.zeros(n)
    for i in range(n):
        if X[i] == 1:
            y_i = np.random.binomial(1, p)
        elif X[i] == 0:
            y_i = np.random.binomial(1, q)
        else:
            raise ValueError("incorrect value of X")
        Y[i] = y_i
    return X, Y


def compute_empirical_risk(f, X, Y):
    """
        Compute empirical risk of predictor
        on the dataset

        Parameters:
            X: 1D array
            Y: 1D array
            f: predictor

        Returns:
            empirical risk

        We use the "0-1"-loss
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples)
    for i in range(n_samples):
        predictions[i] = f(X[i])
    return (predictions!=Y).sum()/n_samples


p = 1/3
q = 3/4
max_n_samples = 1500

generalization_error_f_1 = (1-p)/2+q/2
generalization_error_f_2 = 1-generalization_error_f_1
generalization_error_f_3 = (1-p)/2+(1-q)/2


# test f1
empirical_risks = list()
for n in range(1, max_n_samples):
    X, Y = sample_dataset(n, p, q)
    empirical_risks.append(compute_empirical_risk(f1, X, Y))
plt.plot(range(1, max_n_samples), empirical_risks, "o", markersize=2, alpha=0.3, label=r"$R_n(f)$"+" empirical risk")
plt.plot(range(1, max_n_samples), (max_n_samples-1)*[generalization_error_f_1], color="hotpink",label="real risk / generalization error")
plt.xlabel("n")
plt.legend(loc="best")
plt.title(r"$f_1$"+": Empirical risk and generalization error"+"\n"+f"$R(f_1)$"+f"={generalization_error_f_1:.2f}")
plt.savefig("empirical_risk_and_generalization_error_f1.pdf")
plt.close()

# test f2
empirical_risks = list()
for n in range(1, max_n_samples):
    X, Y = sample_dataset(n, p, q)
    empirical_risks.append(compute_empirical_risk(f2, X, Y))
plt.plot(range(1, max_n_samples), empirical_risks, "o", markersize=2, alpha=0.3, label=r"$R_n(f)$"+" empirical risk")
plt.plot(range(1, max_n_samples), (max_n_samples-1)*[generalization_error_f_2], color="hotpink",label="real risk / generalization error")
plt.xlabel("n")
plt.legend(loc="best")
plt.title(r"$f_2$"+": Empirical risk and generalization error"+"\n"+f"$R(f_2)$"+f"={generalization_error_f_2:.2f}")
plt.savefig("empirical_risk_and_generalization_error_f2.pdf")
plt.close()

# test f3
empirical_risks = list()
for n in range(1, max_n_samples):
    X, Y = sample_dataset(n, p, q)
    empirical_risks.append(compute_empirical_risk(f3, X, Y))
plt.plot(range(1, max_n_samples), empirical_risks, "o", markersize=2, alpha=0.3, label=r"$R_n(f)$"+" empirical risk")
plt.plot(range(1, max_n_samples), (max_n_samples-1)*[generalization_error_f_3], color="hotpink",label="real risk / generalization error")
plt.xlabel("n")
plt.legend(loc="best")
plt.title(r"$f_3$"+": Empirical risk and generalization error"+"\n"+f"$R(f_3)$"+f"={generalization_error_f_3:.2f}")
plt.savefig("empirical_risk_and_generalization_error_f3.pdf")
