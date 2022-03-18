import numpy as np

n_list = [200]
# d_list = [10, 15, 20, 25, 30]
# d_list = [n*10 for n in range(5, 11)]
d_list = [n*10 for n in range(10, 21)]
# d_list = [10, 15, 20, 25, 30]

for n in n_list:
    for d in d_list:
        # X = np.random.rand(n, d)
        sigma = 0.02
        X = np.random.rand(n, d-1)
        X_last_column = X[:, -1].reshape(n, 1)
        noise = np.random.normal(0, sigma, size=(X_last_column.shape))
        X_added_column = X_last_column + noise
        X = np.hstack((X, X_added_column))
        # np.save("data/design_matrix", X)
        np.save(f"data/design_matrix_n={n}_d={d}", X)
