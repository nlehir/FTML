import torch
import matplotlib.pyplot as plt

setting = "underparametrized"
# setting = "overparametrized"

# load data
X = torch.load("data/X_" + setting)
y = torch.load("data/y_" + setting)

batch = X.shape[0]
dim_in = X.shape[1]
dim_out = y.shape[1]
# overparametrized
dim_h = 80
print(f"batch: {batch}")
print(f"dim_in: {dim_in}")
print(f"dim_h: {dim_h}")
print(f"dim_out: {dim_out}")

loss_fn = torch.nn.MSELoss(reduction="sum")
n_iterations = int(1e5)
nb_stored_steps = 200
stored_iterations = [
    k * (n_iterations / nb_stored_steps) for k in range(nb_stored_steps)
]


def test_learning_rate(learning_rate: float) -> list[float]:
    """
    Train a neural network with one hidden layer 
    by SGD with a constant learning rate.
    """
    print(f"\ntest learning rate {learning_rate:.3E}")
    Neural_network = torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_h), torch.nn.ReLU(), torch.nn.Linear(dim_h, dim_out)
    )
    losses = list()
    pred_y = Neural_network(X)
    optim = torch.optim.SGD(Neural_network.parameters(), lr=learning_rate, momentum=0.9)
    for iteration in range(n_iterations):
        pred_y = Neural_network(X)
        loss = loss_fn(pred_y, y)
        # SGD update
        optim.zero_grad()
        loss.backward()
        optim.step()
        if iteration in stored_iterations:
            print(f"iteration: {iteration},  loss: {loss:.3E}")
            losses.append(loss.item())
    return losses


# compare several learning rates
learning_rates = [0.005] + [10 ** (-k) for k in range(3, 6)]
for learning_rate in learning_rates:
    losses = test_learning_rate(learning_rate)
    plt.plot(
        stored_iterations,
        losses,
        "o",
        label=r"$\gamma=$" + f"{learning_rate}",
        markersize=3,
        alpha=0.6,
    )
plt.xlabel("iteration")
plt.ylabel("loss")
plt.xscale("log")
plt.yscale("log")
plt.legend(loc="best")
title = (
    "learning curves: SGD, one hidden layer NN\n"
    + setting
    + f"\ninput dim: {dim_in}, batch size: {batch}"
    + f"\nhidden dim: {dim_h}"
    + f"\noutput dim: {dim_out}"
)
plt.title(title)
plt.tight_layout()
plt.savefig("learning_rates_" + setting + ".jpg")
