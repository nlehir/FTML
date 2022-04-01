"""
    perform PCA using sklearn
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# load and center the data
pca_data = np.load("Exercices_5_data.npy")
x_data = pca_data[:, 0]
y_data = pca_data[:, 1]
x_mean = np.mean(x_data)
y_mean = np.mean(y_data)
# center the data
x_data = x_data-np.mean(x_data)
y_data = y_data-np.mean(y_data)
pca_data = np.column_stack((x_data, y_data))
# plot the data
plt.scatter(x_data, y_data)
plt.axis('equal')
plt.title("centered data")
plt.savefig("Exercise_5_centered_data.pdf")

# load the sklearn estimator
pca = PCA(n_components=2)
pca.fit(pca_data)

# principal component obtained by the algorithm
print("components")
print(pca.components_)

# variance carried by those axes
print(f"\nexplained variance {pca.explained_variance_}")

# variance ratio carried by those axes
print(f"\nexplained variance ratio {pca.explained_variance_ratio_}")
