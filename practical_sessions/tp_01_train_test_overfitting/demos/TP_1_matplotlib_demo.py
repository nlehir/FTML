import matplotlib.pyplot as plt
import numpy as np

xmin = -5
xmax = 5
x_data_1 = np.linspace(xmin, xmax, 100)
y_data_1 = x_data_1**2
plt.xlim = (-6, 6)
plt.ylim = (-1, 30)
plt.plot(x_data_1, y_data_1, alpha=0.7, color="aqua", label="data 1")

x_data_2 = np.random.uniform(-4, 4, size=(100))
y_data_2 = x_data_2**2 + np.random.normal(0, 1, size=x_data_2.shape)
plt.plot(x_data_2, y_data_2, "o", markersize=3, alpha=0.9, label="data 2")

title = "matplitlib demo"
plt.gca()
plt.title(title)
plt.savefig("matplotlib_demo.pdf")
