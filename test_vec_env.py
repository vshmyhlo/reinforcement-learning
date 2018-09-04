import numpy as np
from vec_env import VecEnv


class SampleEnv(object):
    class ObservationSpace(object):
        shape = (1,)

    class ActionSpace(object):
        shape = (1,)

        def sample(self):
            return [-1]

    observation_space = ObservationSpace()
    action_space = ActionSpace()

    def reset(self):
        self._i = 0

        return [self._i]

    def step(self, a):
        assert a.shape == self.action_space.shape

        self._i += a[0]

        return [self._i], a[0] * 10, self._i >= 2, None


def test_vec_env():
    env = VecEnv([lambda: SampleEnv() for _ in range(3)])

    s = env.reset()

    assert np.array_equal(s, [[0], [0], [0]])

    a = np.array([[0], [1], [2]])
    s_prime, r, d, _ = env.step(a)

    assert np.array_equal(s_prime, [[0], [1], [2]])
    assert np.array_equal(r, [0, 10, 20])
    assert np.array_equal(d, [False, False, True])

    s = np.where(np.expand_dims(d, -1), env.reset(d), s_prime)

    assert np.array_equal(s, [[0], [1], [0]])

    a = np.array([[0], [1], [2]])
    s_prime, r, d, _ = env.step(a)

    assert np.array_equal(s_prime, [[0], [2], [2]])
    assert np.array_equal(r, [0, 10, 20])
    assert np.array_equal(d, [False, True, True])

    env.close()
