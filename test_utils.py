import impala
import numpy as np
import tensorflow as tf

import utils


class UtilsTest(tf.test.TestCase):
    def test_batch_return(self):
        rewards = [[1, 2, 3]]
        actual = utils.batch_return(rewards, gamma=0.9)
        actual = self.evaluate(actual)
        expected = [[1 + 0.9 * 2 + 0.9 ** 2 * 3, 2 + 0.9 * 3, 3]]

        assert np.allclose(actual, expected)

    def test_batch_n_step_return(self):
        rewards = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        value_prime = [10.0]
        dones = [[False, False, True, False, False, False]]

        actual = utils.batch_n_step_return(rewards, value_prime, dones, gamma=0.9)
        actual = self.evaluate(actual)
        expected = [[2.71, 1.9, 1.0, 10.0, 10.0, 10.0]]

        assert np.allclose(actual, expected)

    def test_from_importance_weights(self):
        log_ratios = np.expand_dims(np.log([0.5, 0.5, 0.5, 2.0, 2.0, 2.0]), 1)
        discounts = np.expand_dims([0.9, 0.9, 0.0, 0.9, 0.9, 0.0], 1)
        rewards = np.expand_dims([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], 1)
        values = np.expand_dims([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], 1)
        value_prime = [100.0]

        actual = impala.from_importance_weights(
            log_ratios, discounts, rewards, values, value_prime
        )
        actual = self.evaluate(actual)

        ratios = np.exp(log_ratios)
        values_prime = np.concatenate([values[1:], np.expand_dims(value_prime, 1)], 0)

        vs_minus_values = np.zeros(values.shape)
        v_minus_value = np.zeros(values.shape[1:])
        for t in reversed(range(vs_minus_values.shape[0])):
            delta = np.minimum(1.0, ratios[t]) * (
                rewards[t] + discounts[t] * values_prime[t] - values[t]
            )
            # v_minus_value += discounts[t] * np.minimum(1., ratios[t]) * delta # TODO: ???
            v_minus_value = delta + discounts[t] * np.minimum(1.0, ratios[t]) * v_minus_value
            vs_minus_values[t] = v_minus_value

        vs = values + vs_minus_values
        vs_prime = np.concatenate([vs[1:], np.expand_dims(value_prime, 0)], 0)
        pg_advantages = np.minimum(1.0, ratios) * (rewards + discounts * vs_prime - values)

        expected = vs, pg_advantages

        assert np.allclose(actual[0], expected[0])
        assert np.allclose(actual[1], expected[1])


def test_episode_tracker():
    s = np.zeros((2,))

    episode_tracker = utils.EpisodeTracker(s)

    assert np.array_equal(episode_tracker.reset(), np.zeros((0, 2)))

    episode_tracker.update([1, 2], np.array([False, False]))
    episode_tracker.update([1, 2], np.array([False, True]))
    episode_tracker.update([1, 2], np.array([True, False]))
    episode_tracker.update([1, 2], np.array([False, True]))

    finished_episodes = episode_tracker.reset()

    assert np.array_equal(
        finished_episodes,
        np.array(
            [
                [2, 4],
                [3, 3],
                [2, 4],
            ]
        ),
    )

    episode_tracker.update([1, 2], np.array([False, False]))
    episode_tracker.update([1, 2], np.array([False, True]))
    episode_tracker.update([1, 2], np.array([True, False]))

    finished_episodes = episode_tracker.reset()

    assert np.array_equal(
        finished_episodes,
        np.array(
            [
                [2, 4],
                [4, 4],
            ]
        ),
    )
