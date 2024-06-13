"""
generate a simple world to study value iteration
"""
import os

import matplotlib.pyplot as plt
import numpy as np

world = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

reward = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


# plot world
plt.imshow(world)
plt.title("1 : available position\n0 : not available position")
plt.colorbar()
figpath = os.path.join("images", "world.pdf")
plt.savefig(figpath)
plt.close()

# plot reward
plt.imshow(reward)
plt.title("reward")
plt.colorbar()
figpath = os.path.join("images", "reward.pdf")
plt.savefig(figpath)
plt.close()

# save world
world_path = os.path.join("data", "world.npy")
np.save(world_path, world)

# save reward
reward_path = os.path.join("data", "reward.npy")
np.save(reward_path, reward)
