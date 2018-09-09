import argparse
import random
import numpy as np
import tensorflow as tf


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def flatten_batch_horizon(array):
    b, h, *shape = array.shape
    return array.reshape((b * h, *shape))


def discounted_reward(rewards, gamma):
    result = np.zeros(rewards.shape[:1])

    for t in reversed(range(rewards.shape[1])):
        result = rewards[:, t] + gamma * result

    return result


# TODO: test
def discounted_return(rewards, gamma):
    returns = np.zeros(rewards.shape)
    ret = 0

    for t in reversed(range(rewards.shape[0])):
        ret = rewards[t] + gamma * ret
        returns[t] = ret

    return returns


def batch_discounted_return(rewards, gamma):
    returns = np.zeros(rewards.shape)
    ret = np.zeros(rewards.shape[:1])

    for t in reversed(range(rewards.shape[1])):
        ret = rewards[:, t] + gamma * ret
        returns[:, t] = ret

    return returns


def batch_return(rewards, gamma, name='batch_return'):
    with tf.name_scope(name):
        value_prime = tf.zeros(tf.shape(rewards)[:1])
        dones = tf.fill(tf.shape(rewards), False)

        return batch_n_step_return(rewards, value_prime, dones, gamma)


def batch_n_step_return(rewards, value_prime, dones, gamma, name='batch_n_step_return'):
    def scan_fn(acc, elem):
        reward, mask = elem

        return reward + mask * gamma * acc

    with tf.name_scope(name):
        rewards, value_prime, dones, gamma = convert_to_tensors(
            [rewards, value_prime, dones, gamma],
            [tf.float32, tf.float32, tf.bool, tf.float32])

        mask = tf.to_float(~dones)
        elems = (tf.transpose(rewards, (1, 0)), tf.transpose(mask, (1, 0)))

        returns = tf.scan(
            scan_fn,
            elems,
            value_prime,
            back_prop=False,
            reverse=True)

        return tf.transpose(returns, (1, 0))


def generalized_advantage_estimation(rewards, values, value_prime, dones, gamma, lam):
    batch_size, horizon = rewards.shape
    gaes = np.zeros((batch_size, horizon))
    gae = np.zeros((batch_size,))
    masks = np.logical_not(dones)

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            delta = rewards[:, t] + gamma * value_prime * masks[:, t] - values[:, t]
        else:
            delta = rewards[:, t] + gamma * values[:, t + 1] * masks[:, t] - values[:, t]

        gae = delta + gamma * lam * masks[:, t] * gae
        gaes[:, t] = gae

    return gaes


def convert_to_tensors(tensors, dtypes):
    assert len(tensors) == len(dtypes)

    return [tf.convert_to_tensor(tensor, dtype) for tensor, dtype in zip(tensors, dtypes)]


def batch_generalized_advantage_estimation(
        rewards, values, value_prime, dones, gamma, lam, name='batch_generalized_advantage_estimation'):
    def scan_fn(acc, elem):
        error, mask = elem

        return error + mask * gamma * lam * acc

    with tf.name_scope(name):
        rewards, values, value_prime, dones, gamma, lam = convert_to_tensors(
            [rewards, values, value_prime, dones, gamma, lam],
            [tf.float32, tf.float32, tf.float32, tf.bool, tf.float32, tf.float32])

        mask = tf.to_float(~dones)
        values_prime = tf.concat([values[:, 1:], tf.expand_dims(value_prime, 1)], 1)
        errors = rewards + mask * gamma * values_prime - values

        elems = (tf.transpose(errors, (1, 0)), tf.transpose(mask, (1, 0)))
        initializer = tf.zeros_like(value_prime)

        gaes = tf.scan(
            scan_fn,
            elems,
            initializer,
            back_prop=False,
            reverse=True)

        return tf.transpose(gaes, (1, 0))


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument('--seed', type=int, default=42)
