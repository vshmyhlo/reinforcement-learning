import gym
import numpy as np


# TODO: buffer rewards
class StackObservation(gym.Wrapper):
    def __init__(self, env, k, dim=-1):
        super().__init__(env)

        self.k = k
        self.dim = dim
        self.buffer = None
        self.observation_space = gym.spaces.Box(
            low=np.expand_dims(self.observation_space.low, -1).repeat(self.k, -1),
            high=np.expand_dims(self.observation_space.high, -1).repeat(self.k, -1),
            dtype=self.observation_space.dtype)

    def reset(self, **kwargs):
        self.buffer = []

        state = self.env.reset()
        self.buffer.append(state)

        while len(self.buffer) < self.k:
            state, reward, done, meta = self.env.step(self.action_space.sample())
            self.buffer.append(state)

        state = np.stack(self.buffer, self.dim)

        return state

    def step(self, action):
        state, reward, done, meta = self.env.step(action)
        self.buffer.append(state)
        self.buffer = self.buffer[-self.k:]

        state = np.stack(self.buffer, self.dim)

        return state, reward, done, meta
