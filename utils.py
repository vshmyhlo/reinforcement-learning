import tensorflow as tf


def select_action_value(value, action):
    num_samples = tf.shape(value)[0]
    indices = tf.stack([tf.range(0, tf.cast(num_samples, action.dtype)), action], -1)

    return tf.gather_nd(value, indices)
