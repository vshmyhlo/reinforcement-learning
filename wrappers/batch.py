import gym
import numpy as np


class Batch(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = np.expand_dims(np.array(obs), 0)

        return obs

    def step(self, action):
        action = np.squeeze(action, 0)
        obs, reward, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()

        obs = np.expand_dims(np.array(obs), 0)
        reward = np.expand_dims(np.array(reward), 0)
        done = np.expand_dims(np.array(done), 0)
        info = [info]

        return obs, reward, done, info

    def render(self, mode="human", index=0, **kwargs):
        assert index == 0

        return super().render(mode, **kwargs)
