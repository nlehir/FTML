"""
    Univariate filtering in order to 
    reduce the vocabulary size and learn an estimator based on a smaller
    vocabulary. In that sense, we have increased the sparsity of the
    estimation.

    Univariate filtering consists in statistically evaluating
    whether a feature is linked to the target variable or not. 
    The features (here the n-grams in the vocabulary) are then 
    ranked according to this degree of correlation.
"""

from utils_data_processing import preprocess_imdb
from sklearn.metrics import make_scorer, accuracy_score
import os
from sklearn.model_selection import GridSearchCV
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ngram_range = (1, 2)
min_df = 2
# Size of the vocabulary to keep.
num_features = 60000
C = 0.5
num_jobs=-1


def sparsity_scorer(clf: Pipeline, *args) -> float:
    """
    Define a sparsity score for the pipeline

    The score mixes the kept vocabulary fraction
    and the sparsity of the estimator.

    The score is a float in [0, 1]
    a pipeline with a 0 score is not sparse at all
    a pipeline with a 1 score is fully sparse
    """
    # count number of kept words, divide by length of vocabulary
    univ_filter = clf.named_steps["filter"]
    mask = univ_filter.get_support()
    nb_used_words = len(np.where(mask)[0])
    fraction_kept_vocabulary = nb_used_words/len(mask)

    # compute sparsiy of the linear estimator
    theta = clf.named_steps["clf"].coef_[0]
    nb_non_null_components = len(np.where(theta)[0])
    fraction_non_null_components_theta = nb_non_null_components/len(theta)

    sparsity = 1-fraction_non_null_components_theta*fraction_kept_vocabulary
    return sparsity


if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=num_jobs)

    filter_function = chi2


    """
    Build a pipeline estimator with 4 steps.

    For the CountVectorizer, it is possible to set
    the binary argument to False or True,
    both approaches make sense and can work well.

    If binary==True, the feature is the
    presence or absence of the n-gram in the vocabulary.

    If binary==False, the feature is the
    number of occurrences of the n-gram in the vocabulary.
    """
    classifier = Pipeline(
        [
            ("vectorizer", CountVectorizer(binary=False, ngram_range=ngram_range, min_df=min_df),),
            ("filter", SelectKBest(filter_function, k=num_features)),
            ("scaler", MaxAbsScaler()),
            ("clf", LogisticRegression(C=C, solver="liblinear")),
        ]
    )

    """
    First learn a Pipeline with a hardcoded number of kept features.
    """
    print("Fit pipeline with hardcoded number of kept features")
    classifier.fit(traindata.data, traindata.target)
    # sparsity = sparsity_scorer(classifier)

    # Extract the selected dimensions
    counter = classifier["vectorizer"]
    chi2_filter = classifier["filter"]
    selected_dims = chi2_filter.get_support()
    selected_scores = chi2_filter.scores_[selected_dims]
    sorted_idx = np.argsort(selected_scores)
    kept_words = np.array(counter.get_feature_names_out())[selected_dims]

    # save vocabulary
    file_name = "vocabulary_univariate_filter.txt"
    print(f"save vocabulary to {file_name}")
    file_path = os.path.join("vocabularies", file_name)
    with open(file_path, "w") as out_file:
        out_file.write("\n".join(kept_words[sorted_idx][::-1]))
    print("Vocabulary saved in vocabularies/vocabulary_univariate_filter.txt")
    nb_kept_words = len(kept_words)
    vocabulary_size=counter.get_feature_names_out().size
    print(f"{selected_dims.sum()} kept words on a vocabulary of size {vocabulary_size}")
    print(f"fraction: {nb_kept_words/vocabulary_size:.2f}")

    acc_train = classifier.score(traindata.data, traindata.target)
    acc_test = classifier.score(testdata.data, testdata.target)
    print(f"train accuracy : {acc_train:.2f}")
    print(f"test accuracy : {acc_test:.2f}")

    """
    Perform a Gridsearch in order to 
    explore the influence of hyperparameters on the sparsity
    and on the accuracy.
    """
    c_values = np.array([0.01, 0.1, 1.0, 2, 5, 10])
    k_values = np.array([1000, 5000, 10000, 15000, 20000, 40000, 60000, 80000])
    param_grid = {"filter__k": k_values, "clf__C": c_values}

    """
    We use a grid search with two scores.
    This is convenient because we can monitor which
    estimator is best for each score separately, and also
    to plot both scores on the same plot.
    """
    grid = GridSearchCV(
        classifier,
        n_jobs=num_jobs,
        param_grid=param_grid,
        scoring={"sparsity": sparsity_scorer, "accuracy": make_scorer(accuracy_score)},
        verbose=2,
        refit=False,
    )

    print("Grid searching")
    grid.fit(traindata.data, traindata.target)

    """
    grid.cv_results_
    is a dict containing various results about the grid search, such as:
    - the fit times
    - the tested parameters
    - the resulting scores for each set of parameters, both for each 
    cross-valisation split and averaged over the folds.
    """
    results = grid.cv_results_

    # Get the index of the estimator with the best test accuracy
    rank = np.array(results["rank_test_accuracy"]).argmin()
    best_pred_metrics = {
        "params": results["params"][rank],
        "sparsity": results["mean_test_sparsity"][rank],
        "mean_accuracy": results["mean_test_accuracy"][rank],
    }
    message = f"Parameters of the estimator with best cross-validated accuracy"
    print(colored(message, "green", attrs=["bold"]))
    print(best_pred_metrics)

    """
    We can use the results of the gridsearch
    in order to plot the two scores as a function of the hyperparameters.
    """
    K, C = np.meshgrid(k_values, c_values)
    mean_sparsity = results["mean_test_sparsity"].reshape((len(c_values), len(k_values)))
    mean_accuracy = results["mean_test_accuracy"].reshape((len(c_values), len(k_values)))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(K, C, mean_sparsity, color="blue", label="sparsity")
    ax.plot_wireframe(K, C, mean_accuracy, color="orange", label="accuracy")
    ax.set_xlabel(r"Number of features kept by $\chi^2$")
    ax.set_ylabel("Inverse of regularization (C)")
    ax.set_zlabel("Sparsity (blue) / Accuracy (orange)")
    # ax.yaxis._set_scale("log")
    plt.xticks(rotation=15)
    plt.legend(loc="best")

    fig_name = "filter.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)
