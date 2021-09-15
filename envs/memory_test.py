import gym
import numpy as np
from gym.envs.registration import register as gym_register


class MemoryTest(gym.Env):
    def __init__(self, n=5, max_steps=6):
        self.observation_space = gym.spaces.Discrete(n)
        self.action_space = gym.spaces.Discrete(n)
        self.max_steps = max_steps

        self.i = None
        self.seq = None
        self.target = None

    def reset(self):
        self.i = 0
        self.seq = np.random.randint(self.observation_space.n, size=[self.max_steps])
        self.target = self.seq[0]

        return self.seq[0]

    def step(self, action):
        self.i += 1

        if self.i < len(self.seq):
            return self.seq[self.i], 0, False, {}
        else:
            return 0, float(action == self.target), True, {}


gym_register(
    id="MemoryTest-v0",
    entry_point="envs:MemoryTest",
)
