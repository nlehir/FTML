import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = np.load("./data/logistic_regression/data.npy")
labels = np.load("./data/logistic_regression/labels.npy")
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
