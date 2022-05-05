import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from TP5_LR_utils import compute_accuracy, visualise_predictions, add_bias, empirical_risk
from TP5_LR_algorithms import GD, SGD, SAG
import os
import cProfile
import sys
from read_params import read_params


data = np.load("./data/logistic_regression/data.npy")
n = data.shape[0]
d = data.shape[1]
print(f"n: {n}")
print(f"d: {d}")
labels = np.load("./data/logistic_regression/labels.npy")
# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.33, random_state=2)
n_train = X_train.shape[0]
n_test = X_test.shape[0]

"""
    Regularization parameter
    enforce strong convexity
"""
params = read_params()
mu = float(params[2])

"""
    Scikit
"""
data = np.load("./data/logistic_regression/data.npy")
labels = np.load("./data/logistic_regression/labels.npy")
# setup profiler
profiler = cProfile.Profile()
profiler.enable()
#
tic = time()
#
clf = LogisticRegression(random_state=0).fit(X_train, y_train.reshape(n_train))
#
toc = time()
scikit_time = toc-tic
# log profile
profiler.disable()
stats_file = open(f"profiling/profiling_scikit.txt", 'w')
sys.stdout = stats_file
profiler.print_stats(sort='time')
sys.stdout = sys.__stdout__
#
theta_scikit = np.append(clf.coef_[0], clf.intercept_)
scikit_test_accuracy = clf.score(X_test, y_test)
scikit_train_accuracy = clf.score(X_train, y_train)
#
# print(f"scikit theta_scikit: {theta_scikit}")
print(f"scikit accuracy train: {scikit_train_accuracy}")
print(f"scikit accuracy test: {scikit_test_accuracy}")
# reshape data to compute empirical risk and perform handwritten GD
X_train = add_bias(X_train)
X_test = add_bias(X_test)
scikit_empirical_risk = empirical_risk(theta_scikit, X_train, y_train)
print(f"scikit empirical risk: {scikit_empirical_risk:.2E}")

max_n_iterations = int(2e4)
# max_n_iterations = int(1e2)

# constants important for the convergence guarantees
# uniform bound on the datapoints
R2 = max(np.linalg.norm(data, axis=1)) ** 2
# smoothness: see TP4.pdf
L = R2
# condition number
kappa = L/mu
print(f"kappa: {kappa:.2E}")
# __import__('ipdb').set_trace()

"""
    GD
"""
theta_gd = np.zeros((d+1, 1))
gamma_GD = 1/L
# gamma_GD = gamma_GD_theory
# gamma_GD = 0.01
theta_gd, empirical_risks_gd, test_errors_gd, gd_time, risk_computations_gd = GD(
    theta_gd, gamma_GD, X_train, y_train, X_test, y_test, max_n_iterations, scikit_empirical_risk, mu)


"""
    SGD
"""
theta_sgd = np.zeros((d+1, 1))
gamma_SGD = 1/L
# gamma_SGD = 0.00001
theta_sgd, empirical_risks_sgd, test_errors_sgd, sgd_time, risk_computations_sgd = SGD(
    theta_sgd, gamma_SGD, X_train, y_train, X_test, y_test, max_n_iterations, scikit_empirical_risk, mu)


"""
    SAG
"""
theta_sag = np.zeros((d+1, 1))
gamma_SAG = 1/(2 * L)
# gamma_SAG = 1/(16 * L)
gamma_SAG = 1/L
# gamma_SAG = gamma_SGD
# gamma_SAG = gamma_GD_theory
# gamma_SAG = 0.00005
theta_sag, empirical_risks_sag, test_errors_sag, sag_time, risk_computations_sag = SAG(
    theta_sag, gamma_SAG, X_train, y_train, X_test, y_test, max_n_iterations, scikit_empirical_risk, mu)


# longest algo
plot_range = risk_computations_gd

title_params = f"n={n}, d={d}\n"+r"$\mu=$"+f"{mu:.2E}\n"+r"$\gamma_{GD}=$"+f"{gamma_GD:.2E}\n" + \
    r"$\gamma_{SGD}=$"+f"{gamma_SGD:.2E}\n" + \
    r"$\gamma_{SAG}=$"+f"{gamma_SAG:.2E}\n"
titlesize = 10
legendsize = 10

"""
    Plot convergence of empirical risk
"""
plt.plot(plot_range, [scikit_empirical_risk for n in range(len(plot_range))],
         label=f"scikit empirical risk after convergence ({scikit_time:.3f}s)")
plt.plot(risk_computations_gd, empirical_risks_gd,
         label=f"GD empirical risk ({gd_time:.3f}s)")
plt.plot(risk_computations_sgd, empirical_risks_sgd,
         label=f"SGD empirical risk ({sgd_time:.3f}s)", alpha=0.7)
plt.plot(risk_computations_sag, empirical_risks_sag,
         label=f"SAG empirical risk ({sag_time:.3f}s)", alpha=0.7)
plt.xlabel("iteration")
plt.ylabel("train loss")
# plt.ylim([-0.1, 1.1])
plt.title(f"Logistic regression: train loss with GD, SGD, SAG, scikit\n" +
          title_params, fontsize=titlesize)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images_LR/convergence.pdf")
plt.close()


"""
    Same plot, but as a function of the number of sample queries
    and in log scale
"""
# add 1 to avoid division by zero in log scale
sample_queries_gd = [d*n_train*i+1 for i in risk_computations_gd]
sample_queries_sgd = [d * i+1 for i in risk_computations_sgd]
sample_queries_sag = [3 * d * i+1 for i in risk_computations_sag]
plot_range_queries = [d * i+1 for i in plot_range]
# compute the difference between empirical risk and scikit risk
diff_risk_gd = [abs(r - scikit_empirical_risk) for r in empirical_risks_gd]
diff_risk_sgd = [abs(r - scikit_empirical_risk) for r in empirical_risks_sgd]
diff_risk_sag = [abs(r - scikit_empirical_risk) for r in empirical_risks_sag]
# plot
plt.plot(np.log10(sample_queries_gd), np.log10(diff_risk_gd),
         label=f"GD empirical risk ({gd_time:.3f}s)", color="orange")
plt.plot(np.log10(sample_queries_sgd), np.log10(diff_risk_sgd),
         label=f"SGD empirical risk ({sgd_time:.3f}s)", alpha=0.7, color="green")
plt.plot(np.log10(sample_queries_sag), np.log10(diff_risk_sag),
         label=f"SAG empirical risk ({sag_time:.3f}s)", alpha=0.7, color="red")
plt.xlabel("log10 number of sample queries (computation time)")
plt.ylabel("log10 excess empirical risk (train loss)")
plt.title(f"Logistic regression: log of train loss with GD, SGD, SAG, scikit\n" +
          title_params, fontsize=titlesize)
plt.legend(loc="best", prop={'size': legendsize})
plt.tight_layout()
plt.savefig("images_LR/convergence_nb_queries.pdf")
plt.close()

"""
    Plot accuracy (test error)
"""
plt.plot(plot_range, [scikit_test_accuracy for n in range(
    len(plot_range))], label="scikit test accuracy after convergence")
plt.plot(risk_computations_gd, test_errors_gd, label="GD test accuracy")
plt.plot(risk_computations_sgd, test_errors_sgd, label="SGD test accuracy")
plt.plot(risk_computations_sag, test_errors_sag, label="sag test accuracy")
plt.xlabel("iteration")
plt.ylabel("test accuracy")
plt.ylim([-0.1, 1.1])
plt.title(f"Logistic regression: test accuracy with GD, SGD, SAG, scikit\n" +
          title_params, fontsize=titlesize)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images_LR/accuracy.pdf")
plt.close()


"""
    Same plot, but as a function of the number of sample queries
    and in log scale
"""
# plot
plt.plot(np.log10(sample_queries_gd), [scikit_test_accuracy for n in range(len(
    sample_queries_gd))], label=f"scikit accuracy after convergence ({scikit_time:.3f}s)", alpha=0.7)
plt.plot(np.log10(sample_queries_gd),  test_errors_gd,
         label=f"GD accuracy ({gd_time:.3f}s)", alpha=0.7)
plt.plot(np.log10(sample_queries_sgd), test_errors_sgd,
         label=f"SGD accuracy ({sgd_time:.3f}s)", alpha=0.7)
plt.plot(np.log10(sample_queries_sag), test_errors_sag,
         label=f"SAG accuracy ({sag_time:.3f}s)", alpha=0.7)
plt.xlabel("log10 number of sample queries (computation time)")
plt.ylabel("test accuracy")
plt.ylim([-0.1, 1.1])
plt.title(f"Logistic regression: test accuracy with GD, SGD, SAG, scikit\n" +
          title_params, fontsize=titlesize)
plt.legend(loc="best", prop={'size': legendsize})
plt.tight_layout()
plt.savefig("images_LR/accuracy_nb_queries.pdf")
plt.close()

print(f"\nGD accuracy train: {compute_accuracy(theta_gd, X_train, y_train)}")
print(f"GD accuracy test: {compute_accuracy(theta_gd, X_test, y_test)}")
print(f"GD empirical risk: {empirical_risk(theta_gd, X_train, y_train):.2E}")

print(f"\nSGD accuracy train: {compute_accuracy(theta_sgd, X_train, y_train)}")
print(f"SGD accuracy test: {compute_accuracy(theta_sgd, X_test, y_test)}")
print(f"SGD empirical risk: {empirical_risk(theta_sgd, X_train, y_train):.2E}")

print(f"\nSAG accuracy train: {compute_accuracy(theta_sag, X_train, y_train)}")
print(f"SAG accuracy test: {compute_accuracy(theta_sag, X_test, y_test)}")
print(f"SAG empirical risk: {empirical_risk(theta_sag, X_train, y_train):.2E}")

print(f"\nscikit time: {scikit_time:.4f} seconds")
print(f"GD time: {gd_time:.4f} seconds")
print(f"SGD time: {sgd_time:.4f} seconds ")
print(f"SAG time: {sag_time:.4f} seconds ")
