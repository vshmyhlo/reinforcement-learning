import gym
import numpy as np


class GridWorld(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action = None

    def reset(self, **kwargs):
        self.action = self.action_space.sample()
        return super().reset(**kwargs)

    def step(self, action):
        self.action = action
        return super().step(action)

    def observation(self, obs):
        image = obs['image'][:, :, 0].astype(np.int64)

        return image

        # obs = np.concatenate([
        #     image.reshape(-1),
        #     np.array([self.action], dtype=image.dtype)
        # ])
        #
        # return obs
