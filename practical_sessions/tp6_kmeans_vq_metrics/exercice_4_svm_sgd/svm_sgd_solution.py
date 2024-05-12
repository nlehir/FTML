"""
    Assess the impact of data scaling on the quality of a linear separator
    obtained by SGD.

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from termcolor import colored


def load_data(dataset) -> tuple:
    # load the data
    # The input X are bidimensional
    data_path = os.path.join("data", dataset, "data.npy")
    labels_path = os.path.join("data", dataset, "labels.npy")
    data = np.load(data_path)
    labels = np.load(labels_path)

    # split the set into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.33, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_classifier(
    dataset,
    classifier,
    X_test,
    y_test,
    X_train,
    y_train,
    coefs,
    intercept,
    classifier_name
) -> None:
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


    """
    Plot the found linear separator

    Here, in dimension 2:
    coefs contains a tuple (a_1, a_2)
    and intercept contains b
    such that the separator is obtained with the
    equation
    (a_1, a_2)(x,y) + b = 0

    which means:
    a_1x + a_2y + b> 0

    or (assuming a_2 != 0):
    y = -(b + a_1x)/a_2
    """
    a_1 = coefs[0, 0]
    a_2 = coefs[0, 1]
    b = intercept[0]
    print(f"intercept: {b:.2f}")
    x_data = X_test[:, 0]
    xlim_left = min(x_data)
    xlim_right = max(x_data)
    x_plot = np.linspace(xlim_left, xlim_right, x_data.shape[0])
    y_plot = -(b+a_1*x_plot)/a_2
    plt.plot(x_plot, y_plot, "-", color="green", label="linear separator")
    plt.legend(loc="best")

    # save the results
    plt.axis("tight")
    title = (
        f"{dataset}\n{classifier_name}\n"
        f"score on test set {100*classifier_score:.2f} %"
    )
    plt.title(title)
    figname = f"prediction_test_set_{dataset}_{classifier_name}.pdf"
    figpath = os.path.join("images", figname)
    plt.savefig(figpath)
    plt.close()


def process_dataset(dataset) -> None:
    print(f"\n---\ndataset: {dataset}")
    X_train, X_test, y_train, y_test = load_data(dataset)

    # learn the classifier
    classifier = SGDClassifier(max_iter=1000)
    classifier.fit(X_train, y_train)
    coefs = classifier.coef_
    intercept = classifier.intercept_
    test_classifier(
        dataset,
        classifier,
        X_test,
        y_test,
        X_train,
        y_train,
        coefs,
        intercept,
        "without_standardization",
    )

    # preprocess the data
    # this could also be done using pipelines
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"mean of dataset: {scaler.mean_}")
    print(f"variance of dataset: {scaler.var_}")

    # relearn the classifier on the preprocessed data
    classifier = SGDClassifier(max_iter=1000)
    classifier.fit(X_train, y_train)
    coefs = classifier.coef_
    intercept = classifier.intercept_
    test_classifier(
        dataset,
        classifier,
        X_test,
        y_test,
        X_train,
        y_train,
        coefs,
        intercept,
        "with_standardization",
    )


def main() -> None:
    process_dataset("dataset_1")
    process_dataset("dataset_2")
    process_dataset("dataset_3")


if __name__ == "__main__":
    main()
