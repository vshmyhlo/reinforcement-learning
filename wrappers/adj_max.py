import gym
import numpy as np


class AdjMax(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.obs = None
       
    def reset(self, **kwargs):
        self.obs = self.env.reset()

        obs, reward, done, info = self.env.step(self.action_space.sample())

        obs_max = np.maximum(self.obs, obs)
        self.obs = obs

        return obs_max

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs_max = np.maximum(self.obs, obs)
        self.obs = obs

        return obs_max, reward, done, info
