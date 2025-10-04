"""
Use Vector quantization for classification
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.spatial.distance import cdist
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

N_OPTUNA_TRIALS = 200
SIGMA_MIN = 1e-8
SIGMA_MAX = 1e1

create_directory_if_missing(os.path.join("images"))

"""
Add code here
"""
