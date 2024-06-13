import os

import matplotlib.pyplot as plt
import numpy as np


def plot_position(
    agent_position: tuple[int, int],
    world: np.ndarray,
    step: int,
    image_folder: str,
) -> None:
    """
    plot the agent in its environment
    """
    title = f"position of agent at step {step}"
    world_copy = np.copy(world)
    world_copy[agent_position[0], agent_position[1]] = 3
    plt.imshow(world_copy)
    plt.title(title)
    figpath = os.path.join(image_folder, f"agent_position_step_{step}.pdf")
    plt.savefig(figpath)
    plt.close()


def plot_value_function(
    value_function: np.ndarray,
    step: int,
    image_folder,
) -> None:
    """
    plot the value function while we compute it
    """
    title = f"value function at step {step}"
    plt.imshow(value_function)
    plt.colorbar()
    plt.title(title)
    figpath = os.path.join(image_folder, f"value_function_step_{step}.pdf")
    plt.savefig(figpath)
    plt.close()
