import gym
import numpy as np


# TODO: handle episode end


class StackObs(gym.Wrapper):
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

        obs = self.env.reset()
        self.buffer.append(obs)

        while len(self.buffer) < self.k:
            obs, reward, done, meta = self.env.step(self.action_space.sample())
            assert not done
            self.buffer.append(obs)
           
        obs = np.stack(self.buffer, self.dim)

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.buffer.append(obs)
        self.buffer = self.buffer[-self.k:]

        obs = np.stack(self.buffer, self.dim)

        return obs, reward, done, info
