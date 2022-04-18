import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from TP4_LR_utils import compute_accuracy, visualise_predictions, add_bias, empirical_risk
from TP4_algorithms import GD, SGD, SGA
import os
import cProfile
import sys



data = np.load("./data/logistic_regression/data.npy")
# normalize the data
data = data-data.mean(axis=0)
data = data/data.std(axis=0)
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
stats_file = open(f"profiling_scikit.txt", 'w')
sys.stdout = stats_file
profiler.print_stats(sort='time')
sys.stdout = sys.__stdout__
# 
theta_scikit = np.append(clf.coef_[0], clf.intercept_)
scikit_test_accuracy = clf.score(X_test, y_test)
scikit_train_accuracy = clf.score(X_train, y_train)

# reshape data to compute empirical risk and perform handwritten GD
X_train = add_bias(X_train)
X_test = add_bias(X_test)
scikit_empirical_risk = empirical_risk(theta_scikit, X_train, y_train)
print(f"scikit empirical risk: {scikit_empirical_risk:.2E}")

# Regularization parameter enforce strong convexity
mu = 1

"""
    GD
"""
theta_gd = np.zeros((d+1, 1))
gamma_max = 2/(1/4 * np.linalg.norm(data, 2)**2)
gamma_GD = 0.5
max_n_iterations_gd = int(6e3)
print(f"gamma_max {gamma_max:.2E}")
theta_gd, empirical_risks_gd, test_errors_gd, gd_time, risk_computations_gd = GD(theta_gd, gamma_GD, X_train, y_train, X_test, y_test, max_n_iterations_gd, scikit_empirical_risk, mu)


"""
    SGD
"""
theta_sgd = np.zeros((d+1, 1))
gamma_0 = 0.1
max_n_iterations_sgd = int(2e4)
schedule = "decreasing 2"
theta_sgd, empirical_risks_sgd, test_errors_sgd, sgd_time, risk_computations_sgd = SGD(theta_sgd, gamma_0, X_train, y_train, X_test, y_test, max_n_iterations_sgd, scikit_empirical_risk, mu, schedule)

"""
    SGA
"""
theta_sga = np.zeros((d+1, 1))
max_n_iterations_sga = int(2e4)
theta_sga, empirical_risks_sga, test_errors_sga, sga_time, risk_computations_sga = SGA(theta_sga, gamma_0, X_train, y_train, X_test, y_test, max_n_iterations_sga, scikit_empirical_risk, mu, schedule)

# longest algo
plot_range = risk_computations_gd

title_params = f"n={n}, d={d}\n"+r"$\gamma_{SGD 0}=$"+f"{gamma_0}"+", schedule: "+schedule+"\n"+r"$\gamma_{GD}=$"+f"{gamma_GD}\n"+r"$\mu=$"+f"{mu}"

"""
    Plot convergence of empirical risk
"""
plt.plot(plot_range, [scikit_empirical_risk for n in range(len(plot_range))], label=f"scikit empirical risk after convergence ({scikit_time:.3f}s)")
plt.plot(risk_computations_gd, empirical_risks_gd, label=f"GD empirical risk ({gd_time:.3f}s)")
plt.plot(risk_computations_sgd, empirical_risks_sgd, label=f"SGD empirical risk ({sgd_time:.3f}s)", alpha = 0.7)
plt.plot(risk_computations_sga, empirical_risks_sga, label=f"SGA empirical risk ({sga_time:.3f}s)", alpha = 0.7)
plt.xlabel("iteration (optimization time)")
plt.ylabel("empirical risk")
plt.title(f"Logistic regression: loss with GD, SGD, scikit\n"+title_params)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images_LR/convergence.pdf")
plt.close()

"""
    Same plot, but as a function of the number of sample queries
    and in log scale
"""
# add 1 to avoid division by zero in log scale
sample_queries_gd = [n_train*i+1 for i in risk_computations_gd]
sample_queries_sgd = [i+1 for i in risk_computations_sgd]
sample_queries_sga = [i+1 for i in risk_computations_sga]
plot_range_queries = [i+1 for i in plot_range]
# compute the difference between empirical risk and scikit risk
diff_risk_gd =  [abs(r - scikit_empirical_risk) for r in empirical_risks_gd]
diff_risk_sgd = [abs(r - scikit_empirical_risk) for r in empirical_risks_sgd]
diff_risk_sga = [abs(r - scikit_empirical_risk) for r in empirical_risks_sga]
# plot
plt.plot(np.log10(sample_queries_gd), np.log10(diff_risk_gd), label=f"GD empirical risk ({gd_time:.3f}s)", color="orange")
plt.plot(np.log10(sample_queries_sgd), np.log10(diff_risk_sgd), label=f"SGD empirical risk ({sgd_time:.3f}s)", alpha = 0.7, color = "green")
plt.plot(np.log10(sample_queries_sga), np.log10(diff_risk_sga), label=f"SGA empirical risk ({sga_time:.3f}s)", alpha = 0.7, color="red")
plt.xlabel("log10 number of sample queries (computation time)")
plt.ylabel("log10 excess empirical risk")
plt.title(f"Logistic regression: log excess loss with GD, SGD, scikit\n"+title_params)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images_LR/convergence_nb_queries.pdf")
plt.close()


"""
    Plot accuracy (test error)
"""
plt.plot(plot_range, [scikit_test_accuracy for n in range(len(plot_range))], label="scikit test accuracy after convergence")
plt.plot(risk_computations_gd, test_errors_gd, label="GD test accuracy", alpha = 0.6)
plt.plot(risk_computations_sgd, test_errors_sgd, label="SGD test accuracy", alpha = 0.6)
plt.plot(risk_computations_sga, test_errors_sga, marker="o", label="SGA test accuracy", alpha = 0.6)
plt.xlabel("iteration (optimization time)")
plt.ylabel("accuracy")
plt.ylim([-0.1, 1.1])
plt.title(f"Logistic regression: accuracy with GD, SGD, scikit\n"+title_params)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images_LR/accuracy.pdf")
plt.close()

"""
    Same plot, but as a function of the number of sample queries
    and in log scale
"""
# compute the difference between empirical risk and scikit risk
diff_accuracy_gd = [abs(scikit_test_accuracy-a) for a in test_errors_gd]
diff_accuracy_sgd = [abs(scikit_test_accuracy-a) for a in test_errors_sgd]
diff_accuracy_sga = [abs(scikit_test_accuracy-a) for a in test_errors_sga]
# plot
plt.plot(np.log10(sample_queries_gd), [scikit_test_accuracy for n in range(len(sample_queries_gd))], label=f"scikit accuracy after convergence ({scikit_time:.3f}s)", alpha=0.7)
plt.plot(np.log10(sample_queries_gd),  test_errors_gd, label=f"GD accuracy ({gd_time:.3f}s)", alpha=0.7)
plt.plot(np.log10(sample_queries_sgd), test_errors_sgd, label=f"SGD accuracy ({sgd_time:.3f}s)", alpha = 0.7)
plt.plot(np.log10(sample_queries_sga), test_errors_sga, label=f"SGA accuracy ({sga_time:.3f}s)", alpha = 0.7)
plt.xlabel("log10 number of sample queries (computation time)")
plt.ylabel("log10 of abs diff accuracy")
plt.ylim([-0.1, 1.1])
plt.title(f"Logistic regression: accuracy with GD, SGD, scikit\n"+title_params)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images_LR/accuracy_nb_queries.pdf")
plt.close()



print(f"\nscikit accuracy train: {scikit_train_accuracy:.2f}")
print(f"GD accuracy train: {compute_accuracy(theta_gd, X_train, y_train):.2f}")
print(f"SGD accuracy train: {compute_accuracy(theta_sgd, X_train, y_train):.2f}")

print(f"\nscikit accuracy test: {scikit_test_accuracy:.2f}")
print(f"GD accuracy test: {compute_accuracy(theta_gd, X_test, y_test):.2f}")
print(f"SGD accuracy test: {compute_accuracy(theta_sgd, X_test, y_test):.2f}")

print(f"\nscikit empirical risk: {scikit_empirical_risk:.2f}")
print(f"GD empirical risk: {empirical_risk(theta_gd, X_train, y_train):.2f}")
print(f"SGD empirical risk: {empirical_risk(theta_sgd, X_train, y_train):.2f}")

print(f"\nscikit time: {scikit_time:.4f} seconds")
print(f"GD time: {gd_time:.4f} seconds")
print(f"SGD time: {sgd_time:.4f} seconds ")
