from enum import Enum
from multiprocessing import Pipe, Process

import numpy as np


class VecEnv(object):
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        env = self.envs[0]
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata

    def reset(self):
        state = [env.reset() for env in self.envs]
        state = np.array(state)

        return state

    def step(self, action):
        assert len(action) == len(self.envs)
        state, reward, done, meta = zip(*[env_step(env, a) for env, a in zip(self.envs, action)])

        state = np.array(state)
        reward = np.array(reward)
        done = np.array(done)

        return state, reward, done, meta

    def render(self, mode="human", index=0):
        return self.envs[index].render(mode=mode)

    def seed(self, seed):
        for i, env in enumerate(self.envs):
            env.seed(seed + i)

    def close(self):
        for env in self.envs:
            env.close()


def env_step(env, action):
    state, reward, done, meta = env.step(action)
    if done:
        state = env.reset()
    return state, reward, done, meta
