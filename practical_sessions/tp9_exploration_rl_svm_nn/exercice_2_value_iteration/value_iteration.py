import os
import random

import numpy as np

from plots import plot_position, plot_value_function
from utils import clean, pick_random_position, update_known_rewards, load_data

# set discount factor
GAMMA = 0.8
# set the number of exploration steps
N_STEPS = 200


def move_agent(agent_position: tuple[int, int], world: np.ndarray):
    """
    EDIT THIS FUNCTION

    determine a new random position for the agent,
    in order to continue the exploration of the environment,
    possibly to find rewards at new positions.

    The available next positions could for instance be
    left, right, top, bottom,
    the 4 closest diagonal positions,
    the and also not moving

    Try to use numpy in order to generate a displacement
    with only 1 instruction !
    """
    # boolean representing if we moved the agent
    moved_agent = False
    new_position = agent_position
    return new_position


def update_value_function(
    value_function: np.ndarray,
    known_reward: np.ndarray,
    world: np.ndarray,
) -> np.ndarray:
    """
    EDIT THIS FUNCTION

    Update the value function according to the Bellman equation
    """
    return value_function


def main() -> None:
    world, reward = load_data()

    # initialize quantities
    value_function = np.zeros(world.shape)
    known_reward = np.zeros(world.shape)
    available_positions = np.where(world)[0]
    agent_position = pick_random_position(available_positions)

    # set image folder
    image_folder = os.path.join("images", "value_iteration")
    clean(image_folder)

    # explore the world and update the value function
    for step in range(N_STEPS):
        print(f"step {step} : agent position {agent_position}")
        plot_position(agent_position, world, step, image_folder=image_folder)

        # move the agent randomly
        agent_position = move_agent(agent_position, world)

        known_reward = update_known_rewards(
            reward=reward,
            known_reward=known_reward,
            agent_position=agent_position,
        )

        value_function = update_value_function(
            value_function=value_function,
            known_reward=known_reward,
            world=world,
        )
        plot_value_function(value_function, step, image_folder=image_folder)

        # periodically reinitialize the position of the agent.
        if (step % 15 == 0) and (step > 0):
            print("----- re initialize agent position")
            agent_position = pick_random_position(available_positions)

    # save our evaluation for usage later
    value_function_path = os.path.join("data", "value_function.npy")
    np.save(value_function_path, value_function)


if __name__ == "__main__":
    main()
