import numpy as np
import matplotlib.pyplot as plt

mu_1 = np.array([[1, 2]])
offset = np.array([[4, 4]])
mu_2 = mu_1-offset
std = (1.5, 1.5)

n_samples = 100
half_n_samples = int(n_samples/2)

data_1 = np.random.normal(loc=mu_1, scale=std, size=(half_n_samples, 2))
data_2 = np.random.normal(loc=mu_2, scale=std, size=(half_n_samples, 2))
data = np.vstack((data_1, data_2))
labels_1 = np.ones((half_n_samples, 1))
labels_2 = -np.ones((half_n_samples, 1))
labels = np.vstack((labels_1, labels_2))
np.save("data/logistic_regression/data", data)
np.save("data/logistic_regression/labels", labels)
plt.plot(data_1[:, 0], data_1[:, 1], "o", label="class 1", alpha=0.7)
plt.plot(data_2[:, 0], data_2[:, 1], "o", label="class 2", alpha=0.7)
plt.xlabel("")
plt.title("Data")
plt.legend(loc="best")
plt.savefig("images_LR/classification_data.pdf")
