"""
    Generate an overparametrized problem.
"""

import os
import torch

batch = 20
dim_in = 50
dim_h = 80
dim_out = 1

def main() -> None:
    X = torch.randn(batch, dim_in)
    Neural_network = torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out)
    )
    y = Neural_network(X)

    X_path = os.path.join("data", "X_overparametrized")
    y_path = os.path.join("data", "y_overparametrized")
    torch.save(X, X_path)
    torch.save(y, y_path)

if __name__ == "__main__":
    main()
