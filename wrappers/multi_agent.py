import gym.wrappers
import numpy as np


class MultiAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # TODO: fix this
        self.observation_space = self.observation_space[0]
        self.action_space = self.action_space[0]

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs = np.array(obs)

        return obs

    def step(self, action):
        action = list(action)

        state, reward, done, meta = super().step(action)
        state = np.array(state)
        reward = np.array(reward)
        done = np.array(done)

        # TODO: check
        if np.any(done):
            assert np.all(done)
            state = np.array(super().reset())

        return state, reward, done, meta

    def render(self, mode="human", index=0, **kwargs):
        assert index in [0, 1]

        return super().render(mode, **kwargs)
