"""
    Learn a baseline estimator.

    We build a Pipeline that contains
   - a one-hot encoding of the data
   - a scaling of the data
   - a logistic regression

   The one-hot encoding part has some important parameters, about which
   you can find more info in the doc.
   -  ngram range: choice of the length of the ngrams that are used in the
      CountVectorizer. A possible choice is to use
   the value ngram_range = (1, 2), but you may experiment with other values.
   -  min_df: minimum number of documents or document frequency for a word to be 
   kept in the dicitonary.
"""

from utils_data_processing import preprocess_imdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils import save_vocabulary

from constants import NGRAM_RANGE, MIN_DF, NUM_JOBS




if __name__ == "__main__":
    """
    preprocess_imdb() returns scikit bunches
    For instance,
    - traindata.data contains the list of all source texts.
    - traindata.target contains the list of all binary targets (positive or
      negative review)
    """
    traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)

    # define the pipeline
    clf = Pipeline(
        [
            ("vectorizer", CountVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)),
            ("scaler", MaxAbsScaler()),
            ("clf", LogisticRegression(solver="liblinear")),
        ]
    )

    # fit the classifier
    clf.fit(X=traindata.data, y=traindata.target)

    # save and print info
    file_name = "vocabulary_exercise_1_simple.txt"
    save_vocabulary(clf, file_name=file_name)

    # compute and print the metrics
    acc_train = clf.score(traindata.data, traindata.target)
    acc_test = clf.score(testdata.data, testdata.target)
    print(f"train accuracy : {acc_train:.4f}")
    print(f"test accuracy : {acc_test:.4f}")
