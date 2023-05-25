"""
    Define a target function
    and generate data
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from constants import M_TARGET, N_SAMPLES, SIGMA
from utils import forward_pass

XMIN = -1
XMAX = 1


def main():
    rng = np.random.default_rng()

    scale = 1 / math.sqrt(M_TARGET)

    phi = rng.uniform(-math.pi, math.pi, (1, M_TARGET))
    # wh = 1 / math.sqrt(M_TARGET) * np.column_stack((np.cos(phi), np.sin(phi)))
    wh = np.vstack((np.cos(phi), np.sin(phi)))
    theta = rng.uniform(-scale, scale, M_TARGET + 1)
    # theta = rng.uniform(-2, 2, M_TARGET + 1)

    inputs = np.linspace(XMIN, XMAX, num=N_SAMPLES)
    bayes_predictions = forward_pass(X=inputs, wh=wh, theta=theta)["y_hat"]

    # add noise
    noise = np.random.normal(0, SIGMA, len(inputs))
    outputs = noise + bayes_predictions

    # plot
    plt.plot(inputs, outputs, "o", label="data", alpha=0.8)
    plt.plot(inputs, bayes_predictions, label="bayes predictor", color="aqua")
    plt.xlabel("input")
    plt.ylabel("output")
    title = f"data and Bayes predictor\n" + r"$\sigma=$" + f"{SIGMA}"
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig("data.pdf")
    plt.close()

    # save data
    folder = "data"
    np.save(os.path.join(folder, "inputs"), inputs)
    np.save(os.path.join(folder, "outputs"), outputs)
    np.save(os.path.join(folder, "bayes_predictions"), bayes_predictions)


if __name__ == "__main__":
    main()
