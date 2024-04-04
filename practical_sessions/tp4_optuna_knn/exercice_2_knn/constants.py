import numpy as np

# number of neighbors
k=2

# number of tests for each choice of (n, d)
n_test = 1000

# list of numbers of sample
exponents = range(1, 3)
n_samples_list = [int(10**k) for k in exponents]

# list of dimensions
d_list = [10, 100, 1000]

# initialize PRNG
rng = np.random.default_rng()
