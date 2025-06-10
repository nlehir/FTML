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
from constants import NGRAM_RANGE, MIN_DF, NUM_JOBS
from utils import save_vocabulary

# Size of the vocabulary to keep.
NUMBER_OF_KEPT_FEATURES = 10
C = 0.5

def sparsity_scorer(classifier: Pipeline) -> float:
    """
    Define a sparsity score for the pipeline

    The sparsisity score can for instance be computed
    from the fraction of ketpt words in the vocabulary.
    Optionally and if relevant, it may also include the sparsity of the
    linear estimator.

    The score is a float in [0, 1]
    a pipeline with a 0 score is not sparse at all
    a pipeline with a 1 score is fully sparse

    EDIT THIS FUNCTION
    """
    vectorizer = classifier["vectorizer"]
    chi2_filter = classifier["filter"]
    selected_dims = chi2_filter.get_support()
    kept_words = np.array(vectorizer.get_feature_names_out())[selected_dims]
    nb_kept_words = len(kept_words)
    vocabulary_size=vectorizer.get_feature_names_out().size
    fraction = nb_kept_words / vocabulary_size
    print(f"{selected_dims.sum()} kept words on a vocabulary of size {vocabulary_size}")
    print(f"fraction: {fraction:.4E}")
    sparsity = 1 -  fraction
    return sparsity

def evaluate_k(
        traindata,
        testdata,
        k,
        ):
    """
    Evaluate scores when keeping k features
    """
    classifier = Pipeline(
        [
            ("vectorizer", CountVectorizer(binary=False, ngram_range=NGRAM_RANGE, min_df=MIN_DF),),
            ("filter", SelectKBest(score_func=filter_function, k=k)),
            ("scaler", MaxAbsScaler()),
            ("clf", LogisticRegression(C=C, solver="liblinear")),
        ]
    )

    print("fit")
    classifier.fit(traindata.data, traindata.target)

    # Extract the selected dimensions
    vectorizer = classifier["vectorizer"]
    chi2_filter = classifier["filter"]
    selected_dims = chi2_filter.get_support()
    kept_words = np.array(vectorizer.get_feature_names_out())[selected_dims]

    # save vocabulary
    nb_kept_words = len(kept_words)
    vocabulary_size=vectorizer.get_feature_names_out().size
    print(f"{selected_dims.sum()} kept words on a vocabulary of size {vocabulary_size}")
    print(f"fraction: {nb_kept_words/vocabulary_size:.4E}")

    file_name = f"vocabulary_exercise_2_univariate_k_{k}.txt"
    save_vocabulary(clf=classifier, file_name=file_name, stat_filter=chi2_filter)

    acc_train = classifier.score(traindata.data, traindata.target)
    acc_test = classifier.score(testdata.data, testdata.target)
    print(f"train accuracy : {acc_train:.4f}")
    print(f"test accuracy : {acc_test:.4f}")
    fraction_kept = nb_kept_words / vocabulary_size
    scores = dict(
            acc_train=acc_train,
            acc_test=acc_test,
            fraction_kept=fraction_kept,
            )
    return scores


if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)

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

    """
    Perform a Gridsearch in order to 
    explore the influence of hyperparameters on the sparsity
    and on the accuracy.
    """
    # number of kept features
    k_list = [1, 10, 100, 1000, 10000, 100000]
    acc_train_list = list()
    acc_test_list = list()
    fraction_kept_list = list()
    for k in k_list:
        print(f"\nUnivariate filtering keeping {k} features")
        scores = evaluate_k(
                traindata=traindata,
                testdata=testdata,
                k=k,
                )
        acc_train = scores["acc_train"]
        acc_test = scores["acc_test"]
        fraction_kept = scores["fraction_kept"]
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        fraction_kept_list.append(fraction_kept)

    fraction_list = [k_list]
    plt.plot(fraction_kept_list, acc_train_list, "--r", label="train accuracy")
    plt.xlabel("Fraction of keps features")
    plt.ylabel("Accuracy")
    plt.plot(fraction_kept_list, acc_test_list, label="test accuracy")
    plt.xscale("log")
    plt.legend(loc="best")
    title = (
            "Accuracy of logistic regression after\n"
            "univariate filtering"
            )
    plt.title(title)
    plt.savefig("Univariate_filtering.pdf")
    plt.close()

    """
    Grid search
    """

    classifier = Pipeline(
        [
            ("vectorizer", CountVectorizer(binary=False, ngram_range=NGRAM_RANGE, min_df=MIN_DF),),
            ("filter", SelectKBest(score_func=filter_function, k=NUMBER_OF_KEPT_FEATURES)),
            ("scaler", MaxAbsScaler()),
            ("clf", LogisticRegression(C=C, solver="liblinear")),
        ]
    )

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
        n_jobs=NUM_JOBS,
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
