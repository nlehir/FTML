"""
Recursive feature elimination (RFE) with logistic regression as estimator
"""

import os

import numpy as np
import utils
from constants import MIN_DF, NGRAM_RANGE, NUM_JOBS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from utils_data_processing import LinearPipeline, preprocess_imdb

if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)

    cache_name = "imdb_wrapper.pkz"
    try:
        X_train, y_train, X_test, y_test, vocabulary = utils.load_cache(
            cache_name, ["X_train", "y_train", "X_test", "y_test", "vocabulary"]
        )
    except RuntimeError as err:
        traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)

        print("Vectorizing the data")
        vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)
        X_train = vectorizer.fit_transform(traindata.data)
        y_train = traindata.target
        X_test = vectorizer.transform(testdata.data)
        y_test = testdata.target
        vocabulary = np.array(vectorizer.get_feature_names_out())

        utils.save_cache(
            cache_name,
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "vocabulary": vocabulary,
            },
        )
    print(f"Original vocabulary size : {len(vocabulary)}")

    """
    Add lines here
    """
