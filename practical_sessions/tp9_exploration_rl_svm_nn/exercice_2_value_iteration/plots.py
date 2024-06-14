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


def plot_all(
    agent_position: tuple[int, int],
    world: np.ndarray,
    reward: np.ndarray,
    known_reward: np.ndarray,
    step: int,
    value_function: np.ndarray,
    image_folder: str,
) -> None:
    """
    plot the agent in its environment
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    world_copy = np.copy(world)
    world_copy[agent_position[0], agent_position[1]] = 3
    title = f"position of agent at step {step}"
    axs[0][0].imshow(world_copy)
    axs[0][0].set_title(title)

    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    im1 = axs[0][1].imshow(value_function, vmin=0, vmax=10)
    ax1_divider = make_axes_locatable(axs[0][1])
    # Add an Axes to the right of the main Axes.
    cax1 = ax1_divider.append_axes("right", size="10%", pad="2%")
    cb1 = fig.colorbar(im1, cax=cax1)
    # plt.colorbar(ax=axs[1])
    title = f"value function at step {step}"
    axs[0][1].set_title(title)

    axs[1][0].imshow(reward)
    axs[1][0].set_title("reward")

    axs[1][1].imshow(known_reward)
    axs[1][1].set_title("known reward")


    title = (
            f"Value iteration, step {step}"
            )
    fig.suptitle(title)
    figpath = os.path.join(image_folder, f"value_iteration_step_{step}.pdf")
    plt.tight_layout()
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
