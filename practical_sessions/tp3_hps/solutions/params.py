"""
Set the HP grids to use for grid search.
"""


import json

predictor = "SVC"
if predictor == "SVC":
    rbf_gammas = [10**k for k in range(-6, 6)]
    poly_degrees = [1, 2, 3, 4]
    Cs = [10**k for k in range(-4, 4)]
    grid = [
        {"kernel": ["linear"], "C": Cs},
        {"kernel": ["sigmoid"], "C": Cs},
        {"kernel": ["poly"], "degree": poly_degrees, "C": Cs},
        {"kernel": ["rbf"], "gamma": rbf_gammas, "C": Cs},
    ]
elif predictor == "tree":
    criterions = ["gini", "entropy"]
    splitters = ["best", "random"]
    max_depths = [2, 10, 20, 40, 60, 100]
    min_samples_splits = [5, 10, 20]
    grid = {
        "criterion": criterions,
        "splitter": splitters,
        "max_depth": max_depths,
        "min_samples_split": min_samples_splits,
    }


"""
Display the grid.
"""
print(f"predictor: {predictor}")
print("\nparameter grid:")
for item in grid:
    print(json.dumps(item, indent=4))


"""
Dataset splitting parameters
"""
test_size = 0.33
validation_size = 0.2
n_splits = 5
nb_tests = 20
