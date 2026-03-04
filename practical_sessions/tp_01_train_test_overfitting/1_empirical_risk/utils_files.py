import os

import numpy as np


def clean_filename(name: str):
    weird_characters = [".", " ", ","]
    for character in weird_characters:
        name = name.replace(character, "_")
    return name


def load_data(std: float):
    X_train = np.load(
        os.path.join("data", f"{clean_filename(f'X_train_{std:.1f}')}.npy")
    )
    X_test = np.load(os.path.join("data", f"{clean_filename(f'X_test_{std:.1f}')}.npy"))
    y_train = np.load(
        os.path.join("data", f"{clean_filename(f'y_train_{std:.1f}')}.npy")
    )
    y_test = np.load(os.path.join("data", f"{clean_filename(f'y_test_{std:.1f}')}.npy"))
    return X_train, X_test, y_train, y_test
