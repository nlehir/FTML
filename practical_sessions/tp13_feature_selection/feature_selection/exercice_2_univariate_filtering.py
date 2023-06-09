"""
    Univariate filtering
"""

from pydoc import doc
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
from utils_data_processing import sparsity_scorer

num_jobs = -1

if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=num_jobs)
    """
    Add lines here.
    """
