import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# def add_bias(data):
#     n = data.shape[0]
#     return np.hstack((data, np.ones((n, 1))))
# 
# 
# def logistic_loss(z, y):
#     return np.log(1 + np.exp(-z*y))
# 
# 
# def empirical_risk(theta, X, Y):
#     n_samples = X.shape[0]
#     empirical_risk = 0
#     for i in range(n_samples):
#         x = X[i, :]
#         y = Y[i, :]
#         empirical_risk += logistic_loss(x.dot(theta), y)
#     return 1/n_samples*empirical_risk
# 
# 
# def sigmoid(x):
#     return 1 / (1+np.exp(-x))
# 
# 
# def gradient_one_sample(theta, x, y):
#     return - (x * y * sigmoid(-x.dot(theta) * y)).reshape(d, 1)
# 
# 
# def nablaL(s, y):
#     return -  y * sigmoid(-s * y)
# 
# 
# def gradient(theta, X, Y):
#     # n, d = X.shape
#     # result = np.zeros((d, 1))
#     # for i in range(n):
#     #     x = X[i, :]
#     #     y = Y[i, :]
#     #     result += gradient_one_sample(theta, x, y)
#     # return 1/n*result
#     n = X.shape[0]
#     return 1/n*X.transpose().dot(nablaL(X.dot(theta), Y))
# 
# 
# def sign(x):
#     if x > 0:
#         return 1
#     else:
#         return 0
# 
# 
# def test_error(theta, X, Y):
#     n_samples = Y.shape[0]
#     test_error = 0
#     for i in range(n_samples):
#         x = X[i, :]
#         y = Y[i, :]
#         test_error+=logistic_loss(x.dot(theta), y)
#         # if sign(x.dot(theta)[0]) == sign(y):
#         #     pass
#         # else:
#         #     # print(x.dot(theta)[0])
#         #     # print(y)
#         #     test_error += 1
#     return test_error/n_samples


data = np.load("./data/logistic_regression/data.npy")
labels = np.load("./data/logistic_regression/labels.npy")
# data = add_bias(data)
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.33)
d = data.shape[1]

n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"n train: {n_train}")
print(f"n test: {n_test}")
# plot data
X_train_1 = X_train[np.where(y_train == 1)[0]]
X_train_2 = X_train[np.where(y_train == 0)[0]]
X_test_1 = X_test[np.where(y_test == 1)[0]]
X_test_2 = X_test[np.where(y_test == 0)[0]]
plt.plot(X_train_1[:, 0], X_train_1[:, 1], "o", label="class 1 train", alpha=0.5, color="blue")
plt.plot(X_test_1[:, 0], X_test_1[:, 1], "o", label="class 1 test", alpha=1, color="blue")
plt.plot(X_train_2[:, 0], X_train_2[:, 1], "o", label="class 2 train", alpha=0.5, color="orange")
plt.plot(X_test_2[:, 0], X_test_2[:, 1], "o", label="class 2 test", alpha=1, color="orange")
plt.title("Train set and test set")
plt.legend(loc="best")
plt.savefig("images_LR/train_test.pdf")
plt.close()

theta = np.zeros((d, 1))
tau_max = 2/(1/4 * np.linalg.norm(data, 2)**2)
tau = tau_max
print(tau_max)
tau = 1
n_steps = 1000

# empirical_risks = list()
# test_errors = list()
# for step in range(n_steps):
#     theta -= tau*gradient(theta, X_train, y_train)
#     empirical_risks.append(empirical_risk(theta, X_train, y_train))
#     test_errors.append(test_error(theta, X_test, y_test))
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print("train")
print(clf.score(X_train, y_train))
print("test")
print(clf.score(X_test, y_test))
print("theta")
print(clf.coef_[0])
print("intercept")
print(clf.intercept_)

# plot separator
a_1, a_2 = clf.coef_[0]
b = clf.intercept_
xx = np.linspace(min(X_train[:, 0]), max(X_train[:, 1]))
yy = [(-b-a_1*x)/a_2 for x in xx]
plt.plot(xx, yy, label="separator")
X_train_1 = X_train[np.where(y_train == 1)[0]]
X_train_2 = X_train[np.where(y_train == 0)[0]]
X_test_1 = X_test[np.where(y_test == 1)[0]]
X_test_2 = X_test[np.where(y_test == 0)[0]]
plt.plot(X_train_1[:, 0], X_train_1[:, 1], "o", label="class 1 train", alpha=0.5, color="blue")
plt.plot(X_test_1[:, 0], X_test_1[:, 1], "o", label="class 1 test", alpha=1, color="blue")
plt.plot(X_train_2[:, 0], X_train_2[:, 1], "o", label="class 2 train", alpha=0.5, color="orange")
plt.plot(X_test_2[:, 0], X_test_2[:, 1], "o", label="class 2 test", alpha=1, color="orange")
plt.legend(loc="best")
plt.title("Separation obtained by logistic regression")
plt.savefig("images_LR/separation.pdf")

# plt.plot(range(n_steps), empirical_risks, "o", label="empirical risk", markersize=5, alpha=0.8)
# plt.plot(range(n_steps), test_errors, "o", label="test errors", markersize=5, alpha=0.8)
# plt.xlabel("iteration")
# plt.title("Gradient algorithm for logistic regression")
# plt.legend(loc="best")
# plt.savefig("images_LR/optimisation.pdf")
