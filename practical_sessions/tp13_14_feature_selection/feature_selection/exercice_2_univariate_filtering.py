"""
    Univariate filtering
"""

from utils_data_processing import preprocess_imdb
from sklearn.metrics import make_scorer, accuracy_score
import os
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

num_jobs = -1


def sparsity_scorer(clf: Pipeline, *args) -> float:
    """
    Define a sparsity score for the pipeline

    The score mixes the kept vocabulary fraction
    and the sparsity of the estimator.

    The score is a float in [0, 1]
    a pipeline with a 0 score is not sparse at all
    a pipeline with a 1 score is fully sparse
    """
    # count number of kept words, divide by length of vocabulary
    univ_filter = clf.named_steps["filter"]
    mask = univ_filter.get_support()
    nb_used_words = len(np.where(mask)[0])
    fraction_kept_vocabulary = nb_used_words/len(mask)

    # compute sparsiy of the linear estimator
    theta = clf.named_steps["clf"].coef_[0]
    nb_non_null_components = len(np.where(theta)[0])
    fraction_non_null_components_theta = nb_non_null_components/len(theta)

    sparsity = 1-fraction_non_null_components_theta*fraction_kept_vocabulary
    return sparsity


if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=num_jobs)
    """
    Add lines here.
    """
