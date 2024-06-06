from matplotlib.cbook import _unmultiplied_rgba8888_to_premultiplied_argb32
import utils
from sklearn.base import BaseEstimator
import fetch_dataset_stanford_sentiment
from sklearn.feature_selection._base import SelectorMixin
from joblib import Parallel, delayed
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline



def clean_text(txt: str) -> str:
    """Takes a document (str) as input and apply preprocessing on it :
        - to lowercase
        - remove URLs
        - remove html tags such as <br />
        - remove punctuation
        - remove linebreaks
        - remove words with numbers inside
        - remove words with repeated letters
    """

    # Change to lower case
    txt = txt.lower()

    # Remove links
    url_regexp = "https?://[\w\/\.]+"
    # url_regexp = 'https?://\S+'
    txt = re.sub(url_regexp, " ", txt)

    # Remove the HTML tags <...> (shortest with the '?'
    # character in the regexp)
    # i.e. matches '< br/>' in both '< br/>' and '< br/> >'
    txt = re.sub("<[ a-zA-Z]*/?>", " ", txt)
    # txt = re.sub('<.*?>+', ' ', txt)

    # Remove punctuation
    txt = re.sub("[%s]" % re.escape(string.punctuation), " ", txt)

    # Remove linebreaks
    txt = re.sub("\n", " ", txt)

    # Remove words containing numbers
    txt = re.sub("\w*\d\w*", " ", txt)

    # Remove duplicated characters that are present more than 2 times
    # like : reaaaaaaally => really
    txt = re.sub(r"(.)\1{2,}", r"\1", txt)
    return txt


def preprocess_imdb(num_jobs=-1) -> tuple[list]:
    """Loads and preprocess the IMDB stanford data
    if there is no cache to load.

    Returns
    -------

        traindata, unsupdata, testdata: sklearn.utils.Bunch
                These dictionnaries have a data and target attribute.
                but unsupdata has no target
    """

    cache_name = "imdb.pkz"
    try:
        traindata, unsupdata, testdata = utils.load_cache(
            cache_name, ["traindata", "unsupdata", "testdata"]
        )
        print("Using cached dataset")
    except RuntimeError as _:
        traindata = fetch_dataset_stanford_sentiment.fetch_imdb(subset="train")
        unsupdata = fetch_dataset_stanford_sentiment.fetch_imdb(subset="unsup")
        testdata = fetch_dataset_stanford_sentiment.fetch_imdb(subset="test")

        def preprocess_data(data: list) -> list:
            return Parallel(n_jobs=num_jobs)(delayed(clean_text)(d) for d in data)

        print("Preprocessing the training data")
        traindata.data = preprocess_data(traindata.data)
        print("Preprocessing the test data")
        testdata.data = preprocess_data(testdata.data)
        print("Preprocessing the unsupervised data")
        unsupdata.data = preprocess_data(unsupdata.data)

        """
        Save the data to a cache file, named after the cache_name variable.
        If the preprocessing is changed, this file must be removed
        and the preprocessing must be performed again.
        """
        utils.save_cache(
            cache_name,
            dict(traindata=traindata, unsupdata=unsupdata, testdata=testdata),
        )

    return traindata, unsupdata, testdata


class LinearPipeline(BaseEstimator):
    def __init__(self, pipeline, clf_key):
        super().__init__()
        self.pipeline = pipeline
        self.clf_key = clf_key

    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def transform(self, X):
        return self.pipeline.transform(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)

    @property
    def coef_(self):
        return self.pipeline.named_steps[self.clf_key].coef_
