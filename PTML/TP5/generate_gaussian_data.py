import numpy as np
import matplotlib.pyplot as plt

d = int(30)
mu_1 = np.random.randint(2, 10, d)
mu_2 = np.random.randint(2, 10, d)
std = [7 for k in range(d)]

n_samples = 200
half_n_samples = int(n_samples/2)

print(f"n: {int(n_samples)}")
print(f"d: {d}")

data_1 = np.random.normal(mu_1, scale=std, size=(half_n_samples, d))
data_2 = np.random.normal(mu_2, scale=std, size=(half_n_samples, d))
data = np.vstack((data_1, data_2))
labels_1 = np.ones((half_n_samples, 1))
labels_2 = -np.ones((half_n_samples, 1))
labels = np.vstack((labels_1, labels_2))
np.save("data/logistic_regression/data", data)
np.save("data/logistic_regression/labels", labels)
