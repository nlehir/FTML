import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from TP4_LR_utils import compute_accuracy, visualise_predictions, add_bias, empirical_risk
from TP4_algorithms import GD, SGD, SAG
import os
import cProfile
import sys



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
mu = 1

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
#
# print(f"scikit theta_scikit: {theta_scikit}")
print(f"scikit accuracy train: {scikit_train_accuracy}")
print(f"scikit accuracy test: {scikit_test_accuracy}")
# reshape data to compute empirical risk and perform handwritten GD
X_train = add_bias(X_train)
X_test = add_bias(X_test)
scikit_empirical_risk = empirical_risk(theta_scikit, X_train, y_train)
print(f"scikit empirical risk: {scikit_empirical_risk:.2E}")

max_n_iterations = int(1e4)

"""
    GD
"""
theta_gd = np.zeros((d+1, 1))
gamma_max = 2/(1/4 * np.linalg.norm(data, 2)**2)
gamma = gamma_max
theta_gd, empirical_risks_gd, test_errors_gd, gd_time, risk_computations_gd = GD(theta_gd, gamma, X_train, y_train, X_test, y_test, max_n_iterations, scikit_empirical_risk, mu)


"""
    SGD
"""
# theta_sgd = np.zeros((d+1, 1))
# gamma_max = 2/(1/4 * np.linalg.norm(data, 2)**2)
# gamma = gamma_max
# theta_sgd, empirical_risks_sgd, test_errors_sgd, sgd_time, risk_computations_sgd = SGD(theta_sgd, gamma, X_train, y_train, X_test, y_test, max_n_iterations, scikit_empirical_risk, mu)


"""
    SAG
"""
# theta_sag = np.zeros((d+1, 1))
# R_max = np.amax(np.linalg.norm(data, axis=1))
# gamma_max = 1/(4 * R_max**2)
# theta_sag, empirical_risks_sag, test_errors_sag, sag_time, risk_computations_sag = SAG(theta_sag, gamma, X_train, y_train, X_test, y_test, max_n_iterations, scikit_empirical_risk, mu)


# longest algo
plot_range = risk_computations_gd

"""
    Plot convergence of empirical risk
"""
plt.plot(plot_range, [scikit_empirical_risk for n in range(len(plot_range))], label=f"scikit empirical risk after convergence ({scikit_time:.3f}s)")
plt.plot(risk_computations_gd, empirical_risks_gd, marker = "o", label=f"GD empirical risk ({gd_time:.3f}s)")
# plt.plot(risk_computations_sgd, empirical_risks_sgd, marker = "o", label=f"SGD empirical risk ({sgd_time:.3f}s)", alpha = 0.7)
# plt.plot(risk_computations_sag, empirical_risks_sag, marker = "o", label=f"SAG empirical risk ({sag_time:.3f}s)", alpha = 0.7)
plt.xlabel("iteration")
plt.ylim([-0.1, 1.1])
plt.title(f"Logistic regression: handwritten GD, SGD, scikit\nn:{n}, d:{d}")
plt.legend(loc="best")
plt.savefig("images_LR/convergence.pdf")
plt.close()

"""
    Plot accuracy (test error)
"""
plt.plot(plot_range, [scikit_test_accuracy for n in range(len(plot_range))], label="scikit test accuracy after convergence")
plt.plot(risk_computations_gd, test_errors_gd, label="GD test accuracy")
# plt.plot(risk_computations_sgd, test_errors_sgd, label="SGD test accuracy")
# plt.plot(risk_computations_sag, test_errors_sag, label="sag test accuracy")
plt.xlabel("iteration")
plt.ylim([-0.1, 1.1])
plt.title(f"Logistic regression accuracy: GD, SGD, scikit\nn:{n}, d:{d}")
plt.legend(loc="best")
plt.savefig("images_LR/accuracy.pdf")
plt.close()

print(f"\nGD accuracy train: {compute_accuracy(theta_gd, X_train, y_train)}")
print(f"GD accuracy test: {compute_accuracy(theta_gd, X_test, y_test)}")
print(f"GD empirical risk: {empirical_risk(theta_gd, X_train, y_train):.2E}")

# print(f"\nSGD accuracy train: {compute_accuracy(theta_sgd, X_train, y_train)}")
# print(f"SGD accuracy test: {compute_accuracy(theta_sgd, X_test, y_test)}")
# print(f"SGD empirical risk: {empirical_risk(theta_sgd, X_train, y_train):.2E}")
# 
# print(f"\nSAG accuracy train: {compute_accuracy(theta_sag, X_train, y_train)}")
# print(f"SAG accuracy test: {compute_accuracy(theta_sag, X_test, y_test)}")
# print(f"SAG empirical risk: {empirical_risk(theta_sag, X_train, y_train):.2E}")
# 
# print(f"\nscikit time: {scikit_time:.4f} seconds")
# print(f"GD time: {gd_time:.4f} seconds")
# print(f"SGD time: {sgd_time:.4f} seconds ")
# print(f"SAG time: {sag_time:.4f} seconds ")


"""
    Prepare data plots
"""
X_train_1 = X_train[np.where(y_train == 1)[0]]
X_train_2 = X_train[np.where(y_train == -1)[0]]
X_test_1 = X_test[np.where(y_test == 1)[0]]
X_test_2 = X_test[np.where(y_test == -1)[0]]


"""
    Monitor predictions
"""
visualise_predictions(theta_gd, X_train_1, "blue", "y=1 train")
visualise_predictions(theta_gd, X_train_2, "orange", "y=-1 train")
plt.title("estimations with "+r"$\theta_{GD}$")
plt.legend(loc="best")
plt.savefig("images_LR/estimations_GD.pdf")
plt.close()

visualise_predictions(theta_scikit, X_train_1, "blue", "y=1 train")
visualise_predictions(theta_scikit, X_train_2, "orange", "y=-1 train")
plt.title("estimations with "+r"$\theta_{GD}$")
plt.title("estimations with "+r"$\theta_{scikit}$")
plt.legend(loc="best")
plt.savefig("images_LR/estimations_scikit.pdf")
plt.close()


"""
    Monitor the obtained estimators
"""
plt.plot(X_train_1[:, 0], X_train_1[:, 1], "o", label="y=1 train", alpha=0.5, color="blue", markersize=4)
plt.plot(X_test_1[:, 0], X_test_1[:, 1], "o", label="y=1 test", alpha=1, color="blue", markersize=4)
plt.plot(X_train_2[:, 0], X_train_2[:, 1], "o", label="y=-1 train", alpha=0.5, color="orange", markersize=4)
plt.plot(X_test_2[:, 0], X_test_2[:, 1], "o", label="y=-1 test", alpha=1, color="orange", markersize=4)
color_gd = "green"
color_scikit = "purple"
rescaling = 5
plt.arrow(0, 0, rescaling*theta_gd[0][0], rescaling*theta_gd[1][0],
          width=0.05, label=f"{rescaling}"+r"$\theta$"+" GD", color=color_gd)
plt.arrow(0, 0, rescaling*theta_scikit[0], rescaling*theta_scikit[1],
          width=0.05, label=f"{rescaling}"+r"$\theta$"+" scikit", color=color_scikit)

xx = np.linspace(0.5*min(X_train[:, 0]), 0.5*max(X_train[:, 1]))
# plot separator gd
a_gd_1, a_gd_2, b_gd = theta_gd[0], theta_gd[1], theta_gd[2]
yy_gd = [(-b_gd-a_gd_1*x)/a_gd_2 for x in xx]
plt.plot(xx, yy_gd, label="separator, GD", color=color_gd)
# plot separator scikit
a_scikit_1, a_scikit_2, b_scikit = theta_scikit[0], theta_scikit[1], theta_scikit[2]
yy_scikit = [(-b_scikit-a_scikit_1*x)/a_scikit_2 for x in xx]
plt.plot(xx, yy_scikit, label="separator, scikit", color=color_scikit)

# setup plot
plt.axis('equal')
plt.title("linear separation")
plt.legend(loc="best", prop={"size": 7})
plt.savefig("images_lr/separation.pdf")
plt.close()
