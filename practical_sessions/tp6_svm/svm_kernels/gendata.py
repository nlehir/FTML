import matplotlib.pyplot as plt
import numpy as np

n_samples = 40
d = 2
std = 0.4

mu_1 = np.array([1, 1])
mu_2 = np.array([-1, 1])
mu_3 = np.array([-1, -1])
mu_4 = np.array([1, -1])
data_1 = np.random.normal(mu_1, scale=std, size=(n_samples, d))
data_2 = np.random.normal(mu_2, scale=std, size=(n_samples, d))
data_3 = np.random.normal(mu_3, scale=std, size=(n_samples, d))
data_4 = np.random.normal(mu_4, scale=std, size=(n_samples, d))
data = np.vstack((data_1, data_2, data_3, data_4))
labels_1 = np.ones((n_samples, 1))
labels_3 = np.ones((n_samples, 1))
labels_2 = -np.ones((n_samples, 1))
labels_4 = -np.ones((n_samples, 1))
labels = np.vstack((labels_1, labels_2, labels_3, labels_4))
np.save("./data/data", data)
np.save("./data/labels", labels)

class_1 = data[np.where(labels == 1)[0]]
class_2 = data[np.where(labels == -1)[0]]
plt.plot(class_1[:, 0], class_1[:, 1], "o", label="class 1", alpha=0.7)
plt.plot(class_2[:, 0], class_2[:, 1], "o", label="class 2", alpha=0.7)
plt.xlabel("")
plt.title("Data")
plt.legend(loc="best")
plt.savefig("classification_data.pdf")
