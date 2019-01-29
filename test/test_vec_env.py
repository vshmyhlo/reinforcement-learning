import numpy as np
from vec_env import VecEnv


class ObservationSpace(object):
    def __init__(self):
        self.shape = (1,)


class ActionSpace(object):
    def __init__(self):
        self.shape = (1,)

    def sample(self):
        return [-1]


class SampleEnv(object):
    def __init__(self):
        self.observation_space = ObservationSpace()
        self.action_space = ActionSpace()
        self.i = None

    def reset(self):
        self.i = 0

        return ['abc'[self.i]]

    def step(self, a):
        assert a.shape == self.action_space.shape

        self.i += a[0]

        return ['abc'[self.i]], a[0] * 10, self.i >= 2, None


def test_vec_env():
    env = VecEnv([lambda: SampleEnv() for _ in range(3)])

    s = env.reset()

    assert np.array_equal(s, [['a'], ['a'], ['a']])

    # step 1
    a = np.array([[0], [1], [2]])
    s_prime, r, d, _ = env.step(a)

    assert np.array_equal(s_prime, [['a'], ['b'], ['a']])
    assert np.array_equal(r, [0, 10, 20])
    assert np.array_equal(d, [False, False, True])

    # step 2
    a = np.array([[0], [1], [2]])
    s_prime, r, d, _ = env.step(a)

    assert np.array_equal(s_prime, [['a'], ['a'], ['a']])
    assert np.array_equal(r, [0, 10, 20])
    assert np.array_equal(d, [False, True, True])

    env.close()
