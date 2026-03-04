import os

import matplotlib.pyplot as plt
import numpy as np
from constants import MEAN_NOISE, N_SAMPLES, STD_NOISE
from sklearn.model_selection import train_test_split
from utils_files import clean_filename


def main():
    print(f"Create dataset with {STD_NOISE} standard deviation, {N_SAMPLES} samples")

    rng = np.random.default_rng()

    def bayes_predictor(x):
        return -3 * x - (x / 2) ** 2 + 500

    # temperature in degree
    temperature = rng.uniform(-5, 35, N_SAMPLES)

    # power consumption in MW
    power_consumption = bayes_predictor(temperature) + rng.normal(
        loc=MEAN_NOISE,
        scale=STD_NOISE,
        size=N_SAMPLES,
    )

    # plot raw dataset
    plt.plot(temperature, power_consumption, "o", alpha=0.7)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Power Consumption (MW)")
    title = (
        "Temperature (°C) vs power consumption (MW)"
        f"\nnoise standard deviation {STD_NOISE:.1f}"
    )
    plt.title(title)
    fig_name = f"dataset_standard_deviation_{STD_NOISE:.1f}"
    fig_name = f"{clean_filename(fig_name)}.pdf"
    fig_path = os.path.join("images", "datasets", fig_name)
    plt.savefig(fig_path)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        temperature,
        power_consumption,
        test_size=0.33,
    )

    # plot dataset with the split
    plt.plot(X_train, y_train, "o", alpha=0.7, label="train set")
    plt.plot(X_test, y_test, "o", alpha=0.7, label="test set")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Power Consumption (MW)")
    title = (
        "Temperature (°C) vs power consumption (MW)"
        f"\nnoise standard deviation {STD_NOISE:.1f}"
    )
    plt.title(title)
    plt.legend(loc="best")
    fig_name = f"dataset_standard_deviation_{STD_NOISE:.1f}_splitted"
    fig_name = f"{clean_filename(fig_name)}.pdf"
    fig_path = os.path.join("images", "datasets", fig_name)
    plt.savefig(fig_path)
    plt.close()

    # save dataset
    np.save(os.path.join("data", clean_filename(f"X_train_{STD_NOISE:.1f}")), X_train)
    np.save(os.path.join("data", clean_filename(f"X_test_{STD_NOISE:.1f}")), X_test)
    np.save(os.path.join("data", clean_filename(f"y_train_{STD_NOISE:.1f}")), y_train)
    np.save(os.path.join("data", clean_filename(f"y_test_{STD_NOISE:.1f}")), y_test)


if __name__ == "__main__":
    main()
