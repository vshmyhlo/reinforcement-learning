import numpy as np
import torch

from utils import n_step_discounted_return
from vec_env import VecEnv


class ThreeStepEnv(object):
    def __init__(self):
        self.observation_space = object()
        self.action_space = object()
        self.i = None

    def reset(self):
        self.i = 0

        return 'abc'[self.i]

    def step(self, a):
        self.i += a

        return 'abc'[self.i], a * 10, self.i >= 2, None

    def close(self):
        pass

    @staticmethod
    def value(s):
        return {
            'a': 20,
            'b': 10,
            'c': 0,
        }[s]


def test_vec_env():
    env = VecEnv([lambda: ThreeStepEnv() for _ in range(3)])

    history = []
    s = env.reset()
    assert np.array_equal(s, ['a', 'a', 'a'])

    # step 1
    a = np.array([0, 1, 2])
    s_prime, r, d, _ = env.step(a)
    history.append((r, d))

    assert np.array_equal(s_prime, ['a', 'b', 'a'])
    assert np.array_equal(r, [0., 10., 20.])
    assert np.array_equal(d, [False, False, True])

    # step 2
    a = np.array([0, 1, 2])
    s_prime, r, d, _ = env.step(a)
    history.append((r, d))

    assert np.array_equal(s_prime, ['a', 'a', 'a'])
    assert np.array_equal(r, [0., 10., 20.])
    assert np.array_equal(d, [False, True, True])

    env.close()

    r, d = zip(*history)
    r = torch.tensor(r, dtype=torch.float).transpose(0, 1)
    d = torch.tensor(d, dtype=torch.bool).transpose(0, 1)

    v = torch.tensor([ThreeStepEnv.value(s) for s in s_prime])
    actual = n_step_discounted_return(r, v, d, 0.9)
    expected = torch.tensor([
        [16.2, 18.],
        [19., 10.],
        [20., 20.],
    ])

    assert torch.allclose(actual, expected)
