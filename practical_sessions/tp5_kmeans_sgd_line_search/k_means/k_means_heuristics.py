"""
    Assess the quality of the clustering of a given dataset using visualizations and heuristics
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from seaborn import pairplot
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer


def main():
    # load the data
    data = np.load("data.npy")
    nbs_of_clusters = range(2, 15)


if __name__ == "__main__":
    main()
