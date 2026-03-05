from sklearn.metrics import mean_squared_error


def mse(
    X,
    theta,
    b,
    y_true,
):
    y_pred = X * theta + b
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    return mse
