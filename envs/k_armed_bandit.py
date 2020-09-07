import gym
import numpy as np


class KArmedBandit(gym.Env):
    def __init__(self, k: int, walk: float = None):
        super().__init__()

        self.k = k
        self.walk = walk
        self.arm_reward = np.random.normal(0, 1, size=(k,))
        self.action_space = gym.spaces.Discrete(k)

    def step(self, action):
        reward = np.random.normal(self.arm_reward[action], 1)

        if self.walk is not None:
            self.arm_reward += np.random.normal(0, self.walk, size=self.arm_reward.shape)
           
        return None, reward, True, None

    def reset(self):
        return None
