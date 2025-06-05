"""
    Download the 50K sentiment analysis dataset from Stanford

    https://ai.stanford.edu/~amaas/data/sentiment/

    25000 train samples
    25000 test samples
"""

import os
import tarfile
import pickle
import shutil
import codecs

from sklearn.datasets._base import RemoteFileMetadata
from sklearn.datasets._base import _fetch_remote, _pkl_filepath
from sklearn.datasets import get_data_home, load_files

ARCHIVE = RemoteFileMetadata(
    filename="aclImdb_v1.tar.gz",
    url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    checksum=("c40f74a18d3b61f90feba1e17730e0d3" "8e8b97c05fde7008942e91923d1658fe"),
)

CACHE_NAME = "imdb.pkz"


def _download_imdb(target_dir, cache_path):
    """Download the Stanford Large Movie Review dataset

    Parameters
    ----------
        target_dir: str
                    The root path where to save the data. Usually, will
                    be ~/scikit_learn_data/imdb_home
        cache_path: str
                    The path of the saved cache file. Usually, will be
                    ~/scikit_learn_data/idmb.pkz
    """

    train_path = os.path.join(target_dir, "aclImdb", "train")
    unsup_path = os.path.join(target_dir, "aclImdb", "unsup")
    test_path = os.path.join(target_dir, "aclImdb", "test")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Downloading dataset from {ARCHIVE.url} (81 MB)")
    archive_path = _fetch_remote(ARCHIVE, dirname=target_dir)

    print(f"Decompressing {archive_path}")
    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)
    os.remove(archive_path)

    # Move the unsup data out of the train subdir
    shutil.move(os.path.join(train_path, "unsup"), os.path.join(unsup_path, "unknown"))

    # Store a zipped pickle
    cache = dict(
        train=load_files(train_path, encoding="latin1"),
        unsup=load_files(unsup_path, encoding="latin1"),
        test=load_files(test_path, encoding="latin1"),
    )
    print("Compressing the dictionnary of the data")
    compressed_content = codecs.encode(pickle.dumps(cache), "zlib_codec")

    print(f"Building up the cache {cache_path}")
    with open(cache_path, "wb") as f:
        f.write(compressed_content)

    shutil.rmtree(target_dir)
    return cache


def fetch_imdb(data_home=None, subset="train", download_if_missing=True):
    """Load the data of the Large Movie Review dataset from stanford

    Download it if necessary.

    Training set : 12500 positive, 12500 negative reviews
    Test set : 12500 positive, 12500 negative reviews
    Unsupervised : 50000 unrated reviews

    Parameters
    ----------

    subset: 'train' or 'test' or 'unsup', optional (default: 'train')
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'unsup' for the unlabeled (not rated) reviews

    download_if_missing : bool, optional (default: True)
       If False, raise an IOError if the data is not locally available
       instead of trying to download the data from the source site.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : list, length [n_samples]
            The data list to learn.
        target: array, shape [n_samples]
            The target labels.
        filenames: list, length [n_samples]
            The path to the location of the data.
        DESCR: str
            The full description of the dataset.
        target_names: list, length [n_classes]
            The names of target classes.

    """
    data_home = get_data_home(data_home=data_home)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    imdb_home = os.path.join(data_home, "imdb_home")
    cache = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, "zlib_codec")
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * "_")
            print("Cache loading failed")
            print(80 * "_")
            print(e)

    if cache is None:
        if download_if_missing:
            print("Downloading Stanford Large Movie Review dataset.")
            cache = _download_imdb(target_dir=imdb_home, cache_path=cache_path)
        else:
            raise IOError("Stanford Large Movie Review dataset not found")

    return cache[subset]


if __name__ == "__main__":
    traindata = fetch_imdb(subset="train")
    unsupdata = fetch_imdb(subset="unsup")
    testdata = fetch_imdb(subset="test")

    print(f"\n{len(traindata.data)} train samples")
    print(f"{len(testdata.data)} test samples")
