import numpy as np
import tensorflow as tf
import utils
import impala


class UtilsTest(tf.test.TestCase):
    def test_batch_return(self):
        rewards = [[1, 2, 3]]
        actual = utils.batch_return(rewards, gamma=0.9)
        actual = self.evaluate(actual)
        expected = [[
            1 + 0.9 * 2 + 0.9**2 * 3,
            2 + 0.9 * 3,
            3
        ]]

        assert np.allclose(actual, expected)

    def test_batch_generalized_advantage_estimation(self):
        rewards = [[1., 1., 1., 1., 1., 1.]]
        values = [[3., 4., 5., 3., 4., 5.]]
        value_prime = [6.]
        dones = [[False, False, True, False, False, False]]

        actual = utils.batch_generalized_advantage_estimation(rewards, values, value_prime, dones, gamma=0.9, lam=0.8)
        actual = self.evaluate(actual)
        expected = [[0.6064, -1.38, -4., 3.40576, 2.508, 1.4]]

        assert np.allclose(actual, expected)

    def test_batch_n_step_return(self):
        rewards = [[1., 1., 1., 1., 1., 1.]]
        value_prime = [10.]
        dones = [[False, False, True, False, False, False]]

        actual = utils.batch_n_step_return(rewards, value_prime, dones, gamma=0.9)
        actual = self.evaluate(actual)
        expected = [[2.71, 1.9, 1., 10., 10., 10.]]

        assert np.allclose(actual, expected)

    def test_from_importance_weights(self):
        log_ratios = np.expand_dims(np.log([0.5, 0.5, 0.5, 2., 2., 2.]), 1)
        discounts = np.expand_dims([0.9, 0.9, 0., 0.9, 0.9, 0.], 1)
        rewards = np.expand_dims([5., 5., 5., 5., 5., 5.], 1)
        values = np.expand_dims([10., 10., 10., 10., 10., 10.], 1)
        value_prime = [100.]

        actual = impala.from_importance_weights(log_ratios, discounts, rewards, values, value_prime)
        actual = self.evaluate(actual)

        ratios = np.exp(log_ratios)
        values_prime = np.concatenate([values[1:], np.expand_dims(value_prime, 1)], 0)

        vs_minus_values = np.zeros(values.shape)
        v_minus_value = np.zeros(values.shape[1:])
        for t in reversed(range(vs_minus_values.shape[0])):
            delta = np.minimum(1., ratios[t]) * (rewards[t] + discounts[t] * values_prime[t] - values[t])
            # v_minus_value += discounts[t] * np.minimum(1., ratios[t]) * delta # TODO: ???
            v_minus_value = delta + discounts[t] * np.minimum(1., ratios[t]) * v_minus_value
            vs_minus_values[t] = v_minus_value

        vs = values + vs_minus_values
        vs_prime = np.concatenate([vs[1:], np.expand_dims(value_prime, 0)], 0)
        pg_advantages = np.minimum(1., ratios) * (rewards + discounts * vs_prime - values)

        expected = vs, pg_advantages

        assert np.allclose(actual[0], expected[0])
        assert np.allclose(actual[1], expected[1])


def test_discounted_reward():
    rewards = np.array([[5, 4, 3]])
    actual = utils.discounted_reward(rewards, 0.9)
    expected = np.array([5 + 0.9 * 4 + 0.9**2 * 3])

    assert np.allclose(actual, expected)
