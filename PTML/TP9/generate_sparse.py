"""
    Generate a regression problem
    with a sparse estimator
"""
import numpy as np

n = 200
d = 100
d_eff = 5
rng = np.random.default_rng()
std_noise = 0.3
theta_star = np.zeros((d, 1))
indexes = np.random.permutation(range(d))
non_null_indexes = indexes[:d_eff]
theta_star[non_null_indexes] = 1
# train
X_train = np.random.uniform(0, 1, (n, d))
noise = rng.normal(0, std_noise, size=(n, 1))
y_train = X_train @ theta_star  + noise
# test
X_test = np.random.uniform(0, 1, (n, d))
noise = rng.normal(0, std_noise, size=(n, 1))
y_test = X_test @ theta_star + noise
# save
np.save("data_regression/X_train", X_train)
np.save("data_regression/y_train", y_train)
np.save("data_regression/X_test", X_test)
np.save("data_regression/y_test", y_test)
print(theta_star)
