"""
    File illustrating the analysis of a time series.
    The time series stores measurements of the level
    of tide as a function of time.
"""

import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

file_name = "data.csv"


"""
Not optimal method to load the data !
It is better to use numpy or pandas direclty.
"""
X = list()
y = list()

with open(file_name, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        X.append(float(row[0]))
        y.append(float(row[1]))


X = np.asarray(X)
y = np.asarray(y)

"""
    Part A : we want to plot the line graph
    of the tide lebel as a function of time.
"""
# plot the line graph of the data
plt.plot(X, y, "o", markersize=4, alpha=0.4)
plt.title("tide level as a function of time")
plt.xlabel("time (hours)")
plt.ylabel("tide level (meters)")
plt.savefig("tide_level.pdf")
plt.close()

"""
    Part D : optimizing a function
"""

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# X_train.sort()
# X_test.sort()
# y_train.sort()
# y_test.sort()


def fit_sinus(X, y):
    """
    function used to fit a sinusoidal function to the data.
    :param X: array of time steps
    :param tide level: array of tide levels
    """
    # guess initial values for the parameters
    # using spectral analysis
    # Fourier transform
    ff = np.fft.fftfreq(len(X), (X[1] - X[0]))
    Fy = abs(np.fft.fft(y))
    guess_freq = abs(ff[np.argmax(Fy[1:]) + 1])
    guess_amp = np.std(y) * 2.0**0.5
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

    # define the function to optimize
    def sinfunc(t, A, w, phi, offset):
        return A * np.sin(w * t + phi) + offset

    popt, _ = scipy.optimize.curve_fit(sinfunc, X, y, p0=guess)

    A, w, phi, offset = popt
    f = w / (2.0 * math.pi)

    def fitted_function(new_time):
        return A * np.sin(w * new_time + phi) + offset

    print(f"amplitude : {A}")
    print(f"period : {1./f}")
    print(f"offset : {offset}")

    return fitted_function


def main():
    # fitted_function = fit_sinus(X_train, y_train)
    fitted_function = fit_sinus(X, y)

    """
        Part E : visually assess our optimized function
    """
    predicted_y_train = fitted_function(X_train)
    predicted_y_test = fitted_function(X_test)

    train_score = r2_score(predicted_y_train, y_train)
    test_score = r2_score(predicted_y_test, y_test)

    plt.plot(
        X_train, y_train, "o", label="train data", alpha=0.6, markersize=4, color="blue"
    )
    plt.plot(
        X_test, y_test, "x", label="test data", alpha=0.6, markersize=4, color="green"
    )
    plt.plot(
        X_train,
        predicted_y_train,
        "o",
        label="model",
        color="orange",
        markersize=1,
        alpha=0.5,
    )
    plt.legend(loc="best")
    plt.title(
        "tide level as a function of time"
        f"\ntrain r2: {train_score:.3E}"
        f"\ntest r2: {test_score:.3E}"
    )
    plt.xlabel("time (hours)")
    plt.ylabel("tide level (meters)")
    plt.tight_layout()
    plt.savefig("prediction.pdf")
    plt.close()


if __name__ == "__main__":
    main()
