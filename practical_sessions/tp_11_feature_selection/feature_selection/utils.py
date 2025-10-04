#!/usr/bin/env python3

"""
Utilitary functions to handle on drive cached data
"""

# Standard modules
import codecs
import logging
import os
import pickle

from sklearn.pipeline import Pipeline


def save_vocabulary(clf: Pipeline, file_name: str, stat_filter=None) -> None:
    """
    Save the vocabulary to a .txt file
    Extract the feature space size.
    """
    vectorizer = clf["vectorizer"]
    vocabulary = vectorizer.get_feature_names_out()
    if stat_filter is not None:
        selected_dims = stat_filter.get_support()
        vocabulary = vocabulary[selected_dims]
    print(f"Feature space size : {len(vocabulary)}")

    file_path = os.path.join("vocabularies", file_name)
    print(f"Saving the vocabulary in {file_name}")
    with open(file_path, "w") as out_file:
        out_file.write("\n".join(vocabulary))


def load_cache(cache_name: str, keys: list[str]) -> tuple:
    """Load cached data from a saved file.

    Parameters
    ----------
        cache_name:  The filepath to the cache file
        keys: A list of keys to fetch in the cache

    Returns
    -------
        data: tuple
             The tuple has as many elements as keys and are the
             the data fetched from the cache
    """
    logger = logging.getLogger("dimred")
    if os.path.exists(cache_name):
        logger.info(f"Loading cache {cache_name}")
        try:
            with open(cache_name, "rb") as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, "zlib_codec")
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * "_")
            print("Cache loading failed")
            print(80 * "_")
            print(e)

        data = ()
        for k in keys:
            data += (cache[k],)
        logger.info(f"Cache {cache_name} loaded")
        return data
    raise RuntimeError(f"Cache {cache_name} not found")


def save_cache(cache_name: str, d: dict) -> None:
    """Save data in a compressed file

    d: a dictionnary with keys the variable names and values the values
       to save. The keys are the ones to provide to load_cache for
       reloading
    """
    logger = logging.getLogger("dimred")
    logger.info(f"Saving the cache {cache_name}")
    compressed_content = codecs.encode(pickle.dumps(d), "zlib_codec")
    with open(cache_name, "wb") as f:
        f.write(compressed_content)
