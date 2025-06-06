"""
    Encourage sparsity by using a L1 regularization (embedded algorithm)
"""

from utils_data_processing import preprocess_imdb
from sklearn.metrics import make_scorer, accuracy_score
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from constants import NUM_JOBS, MIN_DF, NUM_JOBS, NGRAM_RANGE

C = 0.5


def sparsity_scorer(clf: Pipeline, *args) -> float:
    """
    This time the sparsity is only computed from the LogisticRegression theta
    """
    theta = clf.named_steps["clf"].coef_[0]
    nb_non_null_components = len(np.where(theta)[0])
    fraction_non_null_components_theta = nb_non_null_components / len(theta)

    sparsity = 1 - fraction_non_null_components_theta
    return sparsity


if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)

    """
    First learn a Pipeline with a hardcoded regularization parameter C
    """
    classifier = Pipeline(
        [
            ("vectorizer", CountVectorizer(min_df=MIN_DF, ngram_range=NGRAM_RANGE)),
            ("scaler", MaxAbsScaler()),
            ("clf", LogisticRegression(C=C, penalty="l1", solver="liblinear")),
        ]
    )

    classifier.fit(traindata.data, traindata.target)
    sparsity = sparsity_scorer(classifier)
    acc_train = classifier.score(traindata.data, traindata.target)
    acc_test = classifier.score(testdata.data, testdata.target)
    print(f"Sparsity (fraction of zeros in the linear estimator) : {sparsity:.2f}%")
    print(f"train accuracy : {acc_train:.2f}")
    print(f"test accuracy : {acc_test:.2f}")

    # Extract and save the selected vocabulary, saving also the weight
    # corresponding to each word.
    vocabulary = np.array(classifier.named_steps["vectorizer"].get_feature_names_out())
    selected_dims = classifier.named_steps["clf"].coef_.ravel() != 0
    selected_terms = vocabulary[selected_dims]
    weights = classifier.named_steps["clf"].coef_.ravel()[selected_dims]
    sorted_idx = np.argsort(weights)
    print(f"Original vocabulary size : {len(vocabulary)}")
    print(f"Selected vocabulary size : {len(weights)}")

    file_name = "vocabulary_exercise_3_l1.txt"
    print(f"save vocabulary to {file_name}, with weights")
    file_path = os.path.join("vocabularies", file_name)
    with open(file_path, "w") as out_file:
        out_file.write(
            "\n".join(
                [
                    f"{word} ({weight})"
                    for word, weight in zip(
                        selected_terms[sorted_idx], weights[sorted_idx]
                    )
                ]
            )
        )
    print("Selected words : {selected_terms[sorted_idx]}")

    """
    Perform a Gridsearch in order to 
    explore the influence of hyperparameters on the sparsity
    and on the accuracy.
    """
    param_grid = {"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0]}
    grid = GridSearchCV(
        classifier,
        n_jobs=NUM_JOBS,
        param_grid=param_grid,
        scoring={"sparsity": sparsity_scorer, "accuracy": make_scorer(accuracy_score)},
        verbose=1,
        refit=False,
    )
    grid.fit(traindata.data, traindata.target)

    results = grid.cv_results_

    fig, ax1 = plt.subplots()
    ax1.set_title("Sparsity and accuracy of a logistic regression with L1 penalty")
    ax1.set_xlabel("Inverse of the regularization coefficient (C)")
    color = "green"
    ax1.set_ylabel("Mean accuracy on the test folds", color=color)
    ax1.plot(results["param_clf__C"].data, results["mean_test_accuracy"], color=color)
    # ax1.set_xticks(results['param_clf__C'].data.tolist())
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "blue"
    ax2.set_ylabel("Mean sparsity", color=color)
    ax2.plot(results["param_clf__C"].data, results["mean_test_sparsity"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    fig_path = os.path.join("images", "sparsity_l1.pdf")
    plt.savefig(fig_path)
