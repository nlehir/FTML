"""
Example file to illustrate a random dataset
"""

import matplotlib.pyplot as plt
import numpy as np

n_samples = 500
age = np.random.normal(loc=25, scale=6, size=(n_samples, 1))
age = np.maximum(age, 10)
height = np.random.normal(loc=178, scale=9, size=(n_samples, 1))
time = np.random.normal(loc=14, scale=2, size=(n_samples, 1))
time = np.maximum(time, 9.8)

X = np.column_stack((age, height))
np.savetxt("X.txt", X)
np.savetxt("time.txt", time)

plt.hist(age, bins=50)
plt.title("histogram of age")
plt.xlabel("age (years)")
plt.savefig("histogram_age.pdf")
plt.close()

plt.hist(time, bins=50)
plt.title("histogram of 100m time")
plt.xlabel("time (s)")
plt.savefig("histogram_time.pdf")
plt.close()

plt.hist(height, bins=50)
plt.title("histogram of height")
plt.xlabel("height (cm)")
plt.savefig("histogram_height.pdf")
plt.close()
