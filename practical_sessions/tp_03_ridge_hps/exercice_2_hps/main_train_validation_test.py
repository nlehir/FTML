"""
Perform hyperparameter tuning using the train validation test approach and
witness the fact that it is possible to "overfit the validation set",
which means having a validation score that is strictly better than the test
score.
"""

import numpy as np
from data_loading import dataset, load_data, shuffle_data
from params import grid, nb_tests, test_size, validation_size
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_validation_test(X, y, test_size, validation_size) -> tuple[float, float]:
    """
    Compare the test score and the best validation score
    obtained after comparing some hyperparameter sets, only based on
    validation error.
    """

    # randomly split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size
    )

    best_validation_score = 0
    for parameters in ParameterGrid(grid):
        # choose the type of estimator to study
        # classifier = DecisionTreeClassifier()
        classifier = SVC()
        classifier.set_params(**parameters)
        classifier.fit(X_train, y_train)
        validation_score = classifier.score(X_validation, y_validation)
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            best_classifier = classifier
    test_score = best_classifier.score(X_test, y_test)
    return best_validation_score, test_score


def main() -> None:
    """
    Compare validation and test scores
    several times, and average the results in order to
    have a statistically meaningful result.
    """
    validation_scores = list()
    test_scores = list()
    for index in range(nb_tests):
        print(f"simu {index + 1}/{nb_tests}")
        X, y = load_data(dataset)
        X, y = shuffle_data(X, y)
        validation_score, test_score = train_validation_test(
            X=X,
            y=y,
            test_size=test_size,
            validation_size=validation_size,
        )
        validation_scores.append(validation_score)
        test_scores.append(test_score)

    # average and print results
    array_validation_scores = np.asarray(validation_scores)
    array_test_scores = np.asarray(test_scores)
    print("\n------")
    print(f"mean validation score: {array_validation_scores.mean():.3f}")
    print(f"mean test score: {array_test_scores.mean():.3f}")
    print(f"std validation score: {array_validation_scores.std():.3f}")
    print(f"std test score: {array_test_scores.std():.3f}")


if __name__ == "__main__":
    main()
