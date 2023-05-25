"""
    underparametrized problem
"""

import torch

batch = 20
dim_in = 10
dim_h = 5
dim_out = 1

X = torch.randn(batch, dim_in)
Neural_network = torch.nn.Sequential(
    torch.nn.Linear(dim_in, dim_h), torch.nn.ReLU(), torch.nn.Linear(dim_h, dim_out)
)
noise = torch.randn(batch, dim_out)
y = Neural_network(X) + noise

torch.save(X, "data/X_underparametrized")
torch.save(y, "data/y_underparametrized")
