import gym
import numpy as np
from gym.envs.registration import register as gym_register


class SeqCopy(gym.Env):
    def __init__(self, n=5, seq_size=4):
        self.observation_space = gym.spaces.Discrete(n)
        self.action_space = gym.spaces.Discrete(n)
        self.seq_size = seq_size

        self.i = None
        self.seq = None

    def reset(self):
        self.i = 0
        self.seq = np.random.randint(self.observation_space.n - 1, size=[self.seq_size]) + 1
        return self.seq[0]

    def step(self, action):
        first_phase = self.i // self.seq_size == 0

        obs = 0
        reward = 0
        done = False

        if first_phase:
            if action != 0:
                reward -= 1

            if self.i + 1 < self.seq_size:
                obs = self.seq[self.i + 1]
        else:
            i = self.i - self.seq_size
            if action == self.seq[i]:
                reward += 1

            if i + 1 >= self.seq_size:
                done = True

        self.i += 1

        return obs, reward, done, {}


gym_register(
    id="SeqCopy-v0",
    entry_point="envs:SeqCopy",
)
