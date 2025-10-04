"""
Assess the quality of the clustering using the inertia knee criterion
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""

import os

import matplotlib.pyplot as plt
import numpy as np

IMAGE_FOLGER = "images"


def main():
    # load the data
    data = np.load("data.npy")


if __name__ == "__main__":
    main()
