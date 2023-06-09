"""
    Learn a baseline estimator.

    We build a Pipeline that contains
   - a one-hot encoding of the data
   - a scaling of the data
   - a logistic regression

   The one-hot encoding part has some important parameters, about which
   you can find more info in the doc.
   -  ngram range: A possible choice is to use
   the value ngram_range = (1, 2), but you may experiment with other values.
   -  min_df: minimum number of documents or document frequency for a word to be 
   kept in the dicitonary.
"""

from pydoc import doc
from utils_data_processing import preprocess_imdb
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

num_jobs = -1

def save_vocabulary(clf: Pipeline) -> None:
    """
    Save the vocabulary to a .txt file
    Extract the feature space size.
    """
    vectorizer = clf["vectorizer"]
    """
    Add lines here
    """


if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=-1)

    # define the pipeline
    """
    Add lines here
    """
