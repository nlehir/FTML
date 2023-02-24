from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_text(title):
    file_object = open(f"texts/{title}.txt", "r")
    text = file_object.read()
    return text


titles = ["Wim Wenders", "Martin Scorcese", "Karim Benzema", "Antoine Griezmann"]

corpus = [get_text(x) for x in titles]

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)
X_array = X.toarray()


def cosine_simi_func(XX, YY):
    inner_product = np.dot(XX, YY)
    normXX = np.linalg.norm(XX)
    normYY = np.linalg.norm(YY)
    cosine_simi_manual = inner_product / (normXX * normYY)
    return cosine_simi_manual


skl_similarities = cosine_similarity(X_array)

for i in range(len(titles)):
    for j in range(i, len(titles)):
        print(f"{titles[i]} vs {titles[j]}")
        print(f"manual: {cosine_simi_func(X_array[i], X_array[j])}")
        print(f"scikit: {skl_similarities[i, j]}\n")
