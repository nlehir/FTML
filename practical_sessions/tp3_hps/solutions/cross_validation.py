import numpy as np
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC

from data import dataset, load_data, process_data
from params import grid, n_splits, nb_tests, test_size


def cross_validation(X, y, test_size, n_splits) -> None:
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # classifier = DecisionTreeClassifier()
    best_cross_validation_score = 0
    for parameters in ParameterGrid(grid):
        # classifier = DecisionTreeClassifier()
        classifier = SVC()
        classifier.set_params(**parameters)
        kf = KFold(n_splits=n_splits)
        scores = cross_val_score(classifier, X_train, y_train, cv=kf)
        cross_validation_score = scores.mean()
        if cross_validation_score > best_cross_validation_score:
            best_cross_validation_score = cross_validation_score
            best_classifier = classifier
            # stored_scores = scores

    # retrain on complete train set
    best_classifier.fit(X_train, y_train)
    test_score = best_classifier.score(X_test, y_test)

    # score
    # print(f"\nbest classifier train score: {train_score:.3f}")
    # print(f"\nbest classifier scores on folds: {stored_scores}")
    # print(f"best classifier cross validated score: {best_cross_validation_score:.3f}")
    # print(f"best classifier test score: {test_score:.3f}")
    return best_cross_validation_score, test_score


def main() -> None:
    best_cv_scores = list()
    test_scores = list()
    for index in range(nb_tests):
        print(f"simu {index}")
        X, y = load_data(dataset)
        X, y = process_data(X, y)
        best_cross_validation_score, test_score = cross_validation(
            X, y, test_size, n_splits
        )
        best_cv_scores.append(best_cross_validation_score)
        test_scores.append(test_score)

    # average and print results
    array_best_cv_scores = np.asarray(best_cv_scores)
    array_test_scores = np.asarray(test_scores)
    print("\n------")
    print(f"mean best cv score: {array_best_cv_scores.mean():.3f}")
    print(f"mean test score: {array_test_scores.mean():.3f}")
    print(f"std best cv score: {array_best_cv_scores.std():.3f}")
    print(f"std test score: {array_test_scores.std():.3f}")


if __name__ == "__main__":
    main()
