"""
Create data to study overfitting
"""

import matplotlib.pyplot as plt
import numpy as np
from oracle import oracle, sigma
from utils import clean_filename

N_SAMPLES = 100


def main():
    file_name = "./noisy_data.csv"

    inputs = np.random.uniform(0, 1, N_SAMPLES)

    # create data with random noise
    outputs = oracle(inputs) + np.random.normal(0, sigma, inputs.shape)

    # concatenate dataset
    data = np.column_stack((inputs, outputs))

    # save to file
    np.savetxt(file_name, data, delimiter=",")

    title = f"Noisy data: noise std {sigma}\n{N_SAMPLES} samples"
    plt.plot(inputs, outputs, "o", markersize=3)
    plt.xlabel("input")
    plt.ylabel("output")
    plt.title(title)
    fig_name = f"noisy_data_noise_std_{sigma:.3E}_{N_SAMPLES}_samples"
    fig_name = clean_filename(fig_name)
    plt.savefig(f"{fig_name}.pdf")

if __name__ == "__main__":
    main()
