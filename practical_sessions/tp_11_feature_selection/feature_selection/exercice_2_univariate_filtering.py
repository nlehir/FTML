"""
Univariate filtering
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from constants import NUM_JOBS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from utils_data_processing import preprocess_imdb


def sparsity_scorer(clf: Pipeline) -> float:
    """
    Define a sparsity score for the pipeline

    The sparsisity score can for instance be computed
    from the fraction of kept words in the vocabulary.
    Optionally and if relevant, it may also include the sparsity of the
    linear estimator.

    The score is a float in [0, 1]
    a pipeline with a 0 score is not sparse at all
    a pipeline with a 1 score is fully sparse

    EDIT THIS FUNCTION
    """
    sparsity = 0
    return sparsity


if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)
    """
    Add lines here.
    """
