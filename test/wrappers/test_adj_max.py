import gym
import numpy as np

from wrappers.adj_max import AdjMax


class DummyEnv(gym.Env):
    action_space = gym.spaces.Discrete(3)

    def __init__(self):
        self.i = None

    def reset(self):
        self.i = 0

        obs = np.zeros(3)
        obs[self.i % obs.shape[0]] = 1

        return obs

    def step(self, action):
        self.i += 1

        obs = np.zeros(3)
        obs[self.i % obs.shape[0]] = 1

        return obs, None, None, None


def test_adj_max():
    env = AdjMax(DummyEnv())

    obs = env.reset()
    assert np.allclose(
        obs,
        np.array([1., 1., 0.]))

    obs, _, _, _ = env.step(env.action_space.sample())
    assert np.allclose(
        obs,
        np.array([0., 1., 1.]))
