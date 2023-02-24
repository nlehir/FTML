import numpy as np
import os
import matplotlib.pyplot as plt
import math

n_data = 20

# data_1 = np.random.normal(loc=(2, 0.5), scale=0.1, size=(n_data, 2))
# data_2 = np.random.normal(loc=(0, 0), scale=0.5, size=(n_data, 2))
# data = np.concatenate((data_1, data_2))
# np.save("data",data)
data = np.load("./data.npy")

x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, 'bo')


def connectpoints(x, y, p1, p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1, x2], [y1, y2], 'k-')


threshold = 0
distance_type = "L2" # euclidian distance

for (i, j) in [(i, j) for i in range(0, x.shape[0]) for j in range(0, x.shape[0])]:
    x_i = x[i]
    y_i = y[i]
    x_j = x[j]
    y_j = y[j]
    euclidian_distance = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
    if euclidian_distance <= threshold:
        if i is not j:
            connectpoints(x, y, i, j)


plt.title(f"threshold {threshold}\ndistance {distance_type}")
file_name = f"images/thres_{threshold}_dist_{distance_type}.pdf"
plt.savefig(file_name)
plt.close()
