import tensorflow as tf
import utils
import numpy as np


class UtilsTest(tf.test.TestCase):
    def test_select_action_value(self):
        value = tf.convert_to_tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        action = tf.convert_to_tensor([0, 1, 2])

        actual = utils.select_action_value(value, action)
        actual = self.evaluate(actual)

        expected = [1, 5, 9]

        assert np.array_equal(actual, expected)
