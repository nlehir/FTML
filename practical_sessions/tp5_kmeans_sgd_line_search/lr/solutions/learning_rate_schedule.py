import math


def learning_rate_schedule(gamma_0, iteration, schedule) -> float:
    """
    Define the learning rate schedule.
    """
    if schedule == "decreasing 1":
        """
        Theoretical rate for convex losses
        gamma = O(1/t)
        """
        iteration_scale = 100
        return gamma_0 / (1 + iteration / iteration_scale)
    if schedule == "decreasing 2":
        """
        Theoretical rate for strongly convex losses
        It is slightly larger than the previous one
        gamma = O(1/sqr(t))
        """
        iteration_scale = 100
        return gamma_0 / (1 + math.sqrt(iteration / iteration_scale))
    if schedule == "constant":
        """
        Constant learning rate
        """
        return gamma_0
    else:
        raise ValueError("unknown learning rate schedule")
