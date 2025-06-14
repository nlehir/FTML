def generate_output_data(X, theta_star, sigma, r):
    """
    generate input and output data according to
    the linear model, fixed design setup
    - X is fixed
    - y is random, according to

    y = Xtheta_star + epsilon

    where epsilon is a centered gaussian noise vector with variance
    sigma*In

    Parameters:
        X (float matrix): (n, d) design matrix
        theta_star (float vector): (d, 1) vector (optimal parameter)
        sigma (float): variance each epsilon

    Returns:
        y (float matrix): output vector (n, 1)
    """

    # output data
    n = X.shape[0]
    noise = r.normal(0, sigma, size=(n, 1))
    y = X @ theta_star + noise
    return y
