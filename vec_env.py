import itertools
from tqdm import tqdm
import numpy as np


# TODO: refactor

class VecEnv(object):
    def __init__(self, build_env, size):
        self._envs = [build_env(i) for i in range(size)]
        self._action_space = VecActionSpace(self._envs)
        self._observation_space = VecObservationSpace(self._envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self, done=None):
        if done is None:
            state = [env.reset() for env in self._envs]
            state = np.array(state)
        else:
            assert len(done) == len(self._envs)
            state = np.zeros(self.observation_space.shape)

            for i, (env, done) in enumerate(zip(self._envs, done)):
                if done:
                    state[i] = env.reset()

        return state

    def step(self, action):
        assert len(action) == len(self._envs)

        steps = [env.step(action) for env, action in zip(self._envs, action)]
        state, reward, done, meta = zip(*steps)

        state = np.array(state)
        reward = np.array(reward)
        done = np.array(done)

        return state, reward, done, meta


class VecActionSpace(object):
    def __init__(self, envs):
        assert all(env.action_space.n == envs[0].action_space.n for env in envs)

        self._envs = envs
        self._n = envs[0].action_space.n

    @property
    def n(self):
        return self._n

    def sample(self):
        return np.array([env.action_space.sample() for env in self._envs])


class VecObservationSpace(object):
    def __init__(self, envs):
        assert all(env.observation_space.shape == envs[0].observation_space.shape for env in envs)

        self._shape = [len(envs), *envs[0].observation_space.shape]

    @property
    def shape(self):
        return self._shape


def main():
    env = VecEnv('CartPole-v0', size=1)

    s = env.reset()

    for _ in tqdm(itertools.count()):
        a = env.action_space.sample()
        s, r, d, _ = env.step(a)

        s = np.where(np.expand_dims(d, -1), env.reset(d), s)


if __name__ == '__main__':
    main()
