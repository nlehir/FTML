"""
Use Vector quantization for classification
"""


import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import adjusted_rand_score, rand_score

from utils import clean_filename, create_directory_if_missing

sigma = 0.3

"""
https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
"""

X_train = np.load(os.path.join("data", "X_train.npy"), allow_pickle=True)
X_test = np.load(os.path.join("data", "X_test.npy"), allow_pickle=True)
y_train = np.load(os.path.join("data", "y_train.npy"), allow_pickle=True)
y_test = np.load(os.path.join("data", "y_test.npy"), allow_pickle=True)

N_VALIDATION = 5000
VALIDATION_SET = range(N_VALIDATION)
X_VALIDATION = X_test[VALIDATION_SET]
y_validation = y_test[VALIDATION_SET]

N_TEST = 5000
TEST_SET = range(N_VALIDATION, N_VALIDATION + N_TEST)
X_TEST = X_test[TEST_SET]
y_test = y_test[TEST_SET]

N_OPTUNA_TRIALS = 100
SIGMA_MIN = 1e-8
SIGMA_MAX = 1e1

create_directory_if_missing(os.path.join("images"))

# compute the average of each class
averages = np.zeros(shape=(10, 28 * 28))
for d in range(10):
    selected_samples = X_train[y_train == d]
    class_average = np.average(selected_samples, axis=0)
    averages[d] = class_average
    plt.imshow(class_average.reshape(28, 28))
    plt.title(f"Average of class {d}")
    fig_name = f"average_{d}.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)
    plt.close()


print("K means")
print("--------------------")
kmeans = KMeans(n_clusters=10)
kmeans.fit(X=X_train)
for index, proto in enumerate(kmeans.cluster_centers_):
    plt.imshow(proto.reshape(28, 28))
    plt.title(f"K means proto {index}")
    fig_name = f"kmeans_proto_{index}.pdf"
    folder = os.path.join("images", "kmeans")
    fig_path = os.path.join(folder, fig_name)
    create_directory_if_missing(folder)
    plt.savefig(fig_path)
    plt.close()


def test_blur() -> None:
    create_directory_if_missing(os.path.join("images", "test_blur"))
    indexes = range(10)
    sigmas = [0.1, 0.5, 1, 2, 4]
    for index in indexes:
        print(f"print digit {index} from test set")
        # show digit
        digit = X_test[index]
        plt.imshow(digit.reshape(28, 28))
        fig_name = clean_filename(f"digit_{index}")
        fig_path = os.path.join("images", "test_blur", f"{fig_name}.pdf")
        plt.savefig(fig_path)
        plt.close()
        for sigma in sigmas:
            # show blurred with gaussian kernel
            blurred_digit = blur_digit(digit, sigma)
            plt.imshow(blurred_digit)
            fig_name = clean_filename(f"digit_{index}_blurred_{sigma:.2f}")
            fig_path = os.path.join("images", "test_blur", f"{fig_name}.pdf")
            plt.savefig(fig_path)
            plt.close()


def show_errors(
    errors: np.ndarray,
    sigma: float,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> None:
    dir_name = f"errors_{sigma:.3f}"
    dir_name = clean_filename(dir_name)
    dir_path = os.path.join("images", dir_name)
    create_directory_if_missing(dir_path)
    error_indexes = np.where(errors)[0]
    for index in error_indexes:
        digit = X_test[index]
        plt.imshow(digit.reshape(28, 28))
        title = (
            "misclassified digit\n"
            f"label: {labels[index]}\n"
            f"prediction: {predictions[index]}\n"
        )
        plt.title(title)
        plt.tight_layout()
        fig_name = clean_filename(f"digit_{index}")
        fig_path = os.path.join(dir_path, f"{fig_name}.pdf")
        plt.savefig(fig_path)
        plt.close()


def blur_digit(digit: np.ndarray, sigma: float) -> np.ndarray:
    reshaped_digit = digit.reshape(28, 28)
    blurred_digit = cv2.GaussianBlur(
        reshaped_digit, ksize=(57, 57), sigmaX=sigma, sigmaY=sigma
    )
    return blurred_digit


def compute_score_kmeans(
    X: np.ndarray,
    y: np.ndarray,
    kmeans_protos,
) -> float:
    """
    It is not relevant to compute a vanilla
    accuracy, because we do not know the order of the kmeans clusters !
    In other words, there is no connection a priori between the labels and
    the digits.

    This is why other metrics like the rand score are used.
    """
    n_tests = len(X)
    distances = cdist(X, kmeans_protos)
    predictions = np.argmin(distances, axis=1)
    errors = predictions - y
    nb_errors = len(np.where(errors)[0])
    test_accuracy = 1 - nb_errors / n_tests
    print(f"Kmeans test accuracy (not relevant): {test_accuracy:.3f}")
    rand_score_kmeans = rand_score(labels_true=y, labels_pred=predictions)
    adj_rand_score_kmeans = adjusted_rand_score(labels_true=y, labels_pred=predictions)
    print(f"Kmeans test rand_score: {rand_score_kmeans:.3f}")
    print(f"Kmeans test adjusted rand_score: {adj_rand_score_kmeans:.3f}")
    return test_accuracy


def compute_score(X: np.ndarray, y: np.ndarray, sigma: float) -> float:
    n_tests = len(X)

    # process digits and compute pairwise distances
    blurred_digits = np.zeros(shape=(n_tests, 28 * 28))
    # is it possible to do it without a loop ?
    for index, x in enumerate(X):
        blurred_digits[index] = blur_digit(digit=x, sigma=sigma).reshape(1, 28 * 28)
    distances = cdist(blurred_digits, averages)

    # predict
    predictions = np.argmin(distances, axis=1)

    # compute score
    errors = predictions - y
    # show_errors(
    #         errors=errors,
    #         sigma=sigma,
    #         predictions=predictions,
    #         labels=y,
    #         )
    nb_errors = len(np.where(errors)[0])
    test_accuracy = 1 - nb_errors / n_tests
    print(f"VQ test accuracy: {test_accuracy:.3f}")
    return test_accuracy


def objective(trial) -> float:
    sigma = trial.suggest_float("sigma", SIGMA_MIN, SIGMA_MAX, log=True)
    return compute_score(X_VALIDATION, y_validation, sigma)


def main():
    # test_blur()
    storage_name = "sigma.db"
    # clean study if exists
    # you may not want this behavior in practical applications
    if os.path.exists(storage_name):
        os.remove(storage_name)
    # create a study
    # study = optuna.create_study(
    #     storage=f"sqlite:///{storage_name}",
    #     study_name="vector_quantization",
    #     load_if_exists=False,
    #     direction="maximize",
    # )
    # study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    compute_score_kmeans(
        X=X_TEST,
        y=y_test,
        kmeans_protos=kmeans.cluster_centers_,
    )

    # best_sigma = study.best_trial.params["sigma"]

    best_sigma = 0.7

    print("VQ")
    print("--------------------")
    print(f"Best sigma: {best_sigma:.3f}")

    print("\ntest score")
    from time import time

    tic = time()
    print(compute_score(X_TEST, y_test, best_sigma))
    toc = time()
    time_for_n_tests_preds = toc - tic
    time_per_prediction = time_for_n_tests_preds / N_TEST
    print(f"{time_for_n_tests_preds=:.3f} secs")
    print(f"{time_per_prediction=:.3f} secs")

    print("--------------------")
    classifier = LogisticRegression()
    classifier.fit(X=X_train, y=y_train)
    print(f"Logistic regression")
    print(f"train accuracy: {classifier.score(X=X_train, y=y_train)}")
    print(f"test accuracy: {classifier.score(X=X_TEST, y=y_test)}")


if __name__ == "__main__":
    main()
