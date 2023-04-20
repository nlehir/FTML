"""
    We can also use the sklearn library in order to build a linear model of the
    data.
    https://scikit-learn.org/stable/
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from termcolor import colored


def straight_line(a, b, x):
    return a * x + b


def load_data(dataset):
    # load the data
    # The input X are bidimensional
    data = np.load("data/" + dataset + "/data.npy")
    labels = np.load("data/" + dataset + "/labels.npy")

    # split the set into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.33, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_classifier(
    dataset, classifier, X_test, y_test, X_train, y_train, coefs, classifier_name
):
    """
    Function used in order to test the quality of the classifier
    """

    # evaluate the quality of the estimator using the labels
    # on the test set
    classifier_score = classifier.score(X_test, y_test)
    print(classifier_name + colored(f", score={100*classifier_score:.2f} %", "blue"))

    # produce a visualization of the quality of the estimator
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=plt.cm.Paired,
        edgecolor="black",
        label="test set",
        s=20,
    )  # marker size

    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        alpha=0.1,
        cmap=plt.cm.Paired,
        label="training set",
        edgecolor="black",
        s=20,
    )  # marker size

    # plot the found linear separator
    alpha = -coefs[0, 0] / coefs[0, 1]
    # (a,b).(x,y) > 0
    # if and only if
    # ax + by > 0
    # ie:
    # y > - ax/b
    # y > alpha x
    x_data = X_test[:, 0]
    y_data = X_test[:, 1]
    xlim_left = min(x_data)
    xlim_right = max(x_data)
    x_plot = np.linspace(xlim_left, xlim_right, x_data.shape[0])
    y_plot = [straight_line(alpha, 0, x) for x in x_plot]
    plt.plot(x_plot, y_plot, "-", color="green")
    plt.legend(loc="best")

    # save the results
    plt.axis("tight")
    plt.title(
        dataset
        + "\n"
        + classifier_name
        + "\n"
        + f"score on test set {100*classifier_score:.2f} %"
    )
    # plt.tight_layout()
    plt.savefig(
        "images_svm/prediction_test_set_" + dataset + "_" + classifier_name + ".pdf"
    )
    plt.close()


def process_dataset(dataset):
    print("---\ndataset:" + dataset)
    X_train, X_test, y_train, y_test = load_data(dataset)

    # learn the classifier
    classifier = SGDClassifier(max_iter=1000)
    classifier.fit(X_train, y_train)
    coefs = classifier.coef_
    test_classifier(
        dataset,
        classifier,
        X_test,
        y_test,
        X_train,
        y_train,
        coefs,
        "without_standardization",
    )

    # preprocess the data
    # this could also be done using pipelines
    X_total = np.concatenate((X_train, X_test))
    scaler = StandardScaler()
    scaler.fit(X_total)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"mean of dataset: {scaler.mean_}")
    print(f"variance of dataset: {scaler.var_}")

    # relearn the classifier on the preprocessed data
    classifier = SGDClassifier(max_iter=1000)
    classifier.fit(X_train, y_train)
    coefs = classifier.coef_
    test_classifier(
        dataset,
        classifier,
        X_test,
        y_test,
        X_train,
        y_train,
        coefs,
        "with_standardization",
    )


process_dataset("dataset_1")
process_dataset("dataset_2")
process_dataset("dataset_3")
