import gym
import numpy as np


class Batch(gym.Wrapper):
    def reset(self, **kwargs):
        state = self.env.reset()

        state = np.expand_dims(np.array(state), 0)

        return state

    def step(self, action):
        action = np.squeeze(action, 0)

        state, reward, done, meta = self.env.step(action)

        state = np.expand_dims(np.array(state), 0)
        reward = np.expand_dims(np.array(reward), 0)
        done = np.expand_dims(np.array(done), 0)
        meta = [meta]

        return state, reward, done, meta

    def render(self, mode='human', index=0):
        assert index == 0

        return self.env.render(mode=mode)
