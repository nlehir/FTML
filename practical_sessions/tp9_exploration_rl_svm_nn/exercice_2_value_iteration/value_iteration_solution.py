import os

import numpy as np

from plots import plot_all
from utils import clean, pick_random_position, update_known_rewards, load_data

# set discount factor
GAMMA = 0.8
# set the number of exploration steps
N_STEPS = 200
rng = np.random.default_rng()

def move_agent(agent_position: tuple[int, int], world: np.ndarray):
    """
    determine a new random position for the agent.
    in order to continue the exploration of the environment,
    possibly to find rewards at new positions.
    """
    # boolean representing if we moved the agent
    moved_agent = False
    while not moved_agent:
        deplacement = rng.integers(low=-1, high=2, size=2)
        new_position = agent_position + deplacement
        moved_agent = world[new_position[0]][new_position[1]]
    return new_position


def get_next_state_value(i,
                         j,
                         known_reward,
                         value_function,
                         gamma,
                         ):
    return known_reward[i, j] + gamma * value_function[i, j]


def update_value_function(
    value_function: np.ndarray,
    known_reward: np.ndarray,
    world: np.ndarray,
) -> np.ndarray:
    """
    Update the value function according to the Bellman equation
    (perform a sweep)

    Method inspired by the value iteration algorithm
    (see the Sutton book)
    """
    # value_function_copy = np.copy(value_function)
    value_function_copy = value_function
    for i in range(1, world.shape[0] - 1):
        for j in range(1, world.shape[0] - 1):
            # still check that the position is available
            # otherwise, the value should stay at 0
            if world[i, j]:
                next_possible_values = [
                    get_next_state_value(i-1, j, known_reward, value_function_copy, GAMMA),
                    get_next_state_value(i, j-1, known_reward, value_function_copy, GAMMA),
                    get_next_state_value(i, j+1, known_reward, value_function_copy, GAMMA),
                    get_next_state_value(i+1, j, known_reward, value_function_copy, GAMMA),
                    get_next_state_value(i, j, known_reward, value_function_copy, GAMMA),
                ]
                value_function[i, j] = max(next_possible_values)
    return value_function


def main() -> None:
    world, reward = load_data()

    # initialize stuff
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
        # plot_position(agent_position, world, step, image_folder=image_folder)

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
        # plot_value_function(value_function, step, image_folder=image_folder)
        plot_all(
                agent_position=agent_position,
                world=world,
                step=step,
                reward=reward,
                known_reward=known_reward,
                value_function=value_function,
                image_folder=image_folder,
                )

        # periodically reinitialize the position of the agent.
        if (step % 15 == 0) and (step > 0):
            print("----- re initialize agent position")
            agent_position = pick_random_position(available_positions)

    # save our evaluation for usage later
    value_function_path = os.path.join("data", "value_function.npy")
    np.save(value_function_path, value_function)


if __name__ == "__main__":
    main()
