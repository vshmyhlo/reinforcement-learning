import gym
import numpy as np

from vec_env import VecEnv


class Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.i = None

    def reset(self):
        self.i = 0

        return self.i

    def step(self, a):
        self.i += a
        done = self.i >= 10

        return self.i, a * 2, done, None

    def close(self):
        pass


# TODO: better tests
def test_vec_env():
    env = VecEnv([lambda: Env() for _ in range(3)])

    history = []
    s = env.reset()
    assert np.array_equal(s, [0, 0, 0])

    # step 1
    a = np.array([0, 5, 10])
    s_prime, r, d, _ = env.step(a)
    history.append((r, d))

    assert np.array_equal(s_prime, [0, 5, 0])
    assert np.array_equal(r, [0., 10., 20.])
    assert np.array_equal(d, [False, False, True])

    # step 2
    a = np.array([2, 5, 5])
    s_prime, r, d, _ = env.step(a)
    history.append((r, d))

    assert np.array_equal(s_prime, [2, 0, 5])
    assert np.array_equal(r, [4., 10., 10.])
    assert np.array_equal(d, [False, True, False])

    env.close()
