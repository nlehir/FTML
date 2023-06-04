"""
    Generate an underparametrized problem.
"""

import os
import torch

batch = 20
dim_in = 10
dim_h = 5
dim_out = 1

def main() -> None:
    X = torch.randn(batch, dim_in)
    Neural_network = torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out)
    )
    noise = torch.randn(batch, dim_out)
    y = Neural_network(X) + noise

    X_path = os.path.join("data", "X_underparametrized")
    y_path = os.path.join("data", "y_underparametrized")
    torch.save(X, X_path)
    torch.save(y, y_path)

if __name__ == "__main__":
    main()
