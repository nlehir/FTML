"""
Use Vector quantization for classification
"""


import numpy as np
from utils import clean_filename, create_directory_if_missing
import cv2
import matplotlib.pyplot as plt
import optuna
import os
from scipy.spatial.distance import cdist

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

N_OPTUNA_TRIALS = 200
SIGMA_MIN = 1e-6
SIGMA_MAX = 1e0

create_directory_if_missing(os.path.join("images"))

# compute the average of each class
averages = np.zeros(shape=(10, 28 * 28))
for d in range(10):
    selected_samples = X_train[y_train == d]
    class_average = np.average(selected_samples, axis=0)
    averages[d] = class_average
    plt.imshow(class_average.reshape(28, 28))
    fig_name = f"average_{d}.pdf"
    fig_path = os.path.join("images", fig_name)
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


def show_errors(errors: np.ndarray, sigma: float) -> None:
    dir_name = f"errors_{sigma:.3f}"
    dir_name = clean_filename(dir_name)
    dir_path = os.path.join("images", dir_name)
    create_directory_if_missing(dir_path)
    error_indexes = np.where(errors)[0]
    for index in error_indexes:
        digit = X_test[index]
        plt.imshow(digit.reshape(28, 28))
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


def compute_score(X: np.ndarray, y: np.ndarray, sigma: float) -> float:
    n_tests = len(X)

    # process digits and compute pairwise distances
    blurred_digits = np.zeros(shape=(n_tests, 28 * 28))
    for index, x in enumerate(X):
        blurred_digits[index] = blur_digit(x, sigma=sigma).reshape(1, 28 * 28)
    distances = cdist(blurred_digits, averages)

    # predict
    predictions = np.argmin(distances, axis=1)

    # compute score
    errors = predictions - y
    # show_errors(errors, sigma)
    nb_errors = len(np.where(errors)[0])
    mean_accuracy = 1 - nb_errors / n_tests
    print(f"mean accuracy: {mean_accuracy:.3f}")
    return mean_accuracy


def objective(trial) -> float:
    sigma = trial.suggest_float("sigma", SIGMA_MIN, SIGMA_MAX, log=True)
    return compute_score(X_VALIDATION, y_validation, sigma)


def main():
    test_blur()
    storage_name = "sigma.db"
    # clean study if exists
    # you may not want this behavior in practical applications
    if os.path.exists(storage_name):
        os.remove(storage_name)
    # create a study
    study = optuna.create_study(
        storage=f"sqlite:///{storage_name}",
        study_name="vector_quantization",
        load_if_exists=False,
        direction="maximize",
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best_sigma = study.best_trial.params["sigma"]

    print("\ntest score")
    print(compute_score(X_TEST, y_test, best_sigma))


if __name__ == "__main__":
    main()
