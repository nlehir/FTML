import os
import random

import numpy as np


def clean(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))


def pick_random_position(available_positions: np.ndarray) -> tuple[int, int]:
    """
    pick a random position in order to initialize the position of our agent
    """
    number_of_available_positions = available_positions.shape[0]
    random_index = random.randint(0, number_of_available_positions - 1)
    i_coordinate = available_positions[random_index]
    j_coordinate = available_positions[random_index]
    return i_coordinate, j_coordinate


def update_known_rewards(
    reward: np.ndarray,
    known_reward: np.ndarray,
    agent_position: tuple[int, int],
):
    obtained_reward = reward[agent_position[0], agent_position[1]]
    if obtained_reward:
        if not known_reward[agent_position[0], agent_position[1]]:
            print(f"----- found reward at position {agent_position}: {obtained_reward}")
            known_reward[agent_position[0], agent_position[1]] = obtained_reward
    return known_reward


def load_data():
    # load world and reward
    world_path = os.path.join("data", "world.npy")
    world = np.load(world_path)
    reward_path = os.path.join("data", "reward.npy")
    reward = np.load(reward_path)
    return world, reward
