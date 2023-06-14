"""
    Sparsity by regularization (embedded algorithm)
"""

from utils_data_processing import preprocess_imdb
from sklearn.metrics import make_scorer, accuracy_score
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

num_jobs = -1


def sparsity_scorer_nic(clf: Pipeline, *args) -> float:
    """
    This time the sparsity is only computed from the LogisticRegression theta
    """
    theta = clf.named_steps["clf"].coef_[0]
    nb_non_null_components = len(np.where(theta)[0])
    fraction_non_null_components_theta = nb_non_null_components/len(theta)

    sparsity = 1-fraction_non_null_components_theta
    return sparsity


if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=num_jobs)
    """
    Add lines here.
    """
