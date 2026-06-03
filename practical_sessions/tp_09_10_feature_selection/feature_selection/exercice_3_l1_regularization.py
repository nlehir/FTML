"""
Sparsity by regularization (embedded algorithm)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from constants import MIN_DF, NGRAM_RANGE, NUM_JOBS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from utils_data_processing import preprocess_imdb

if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)
    """
    Add lines here.
    """
