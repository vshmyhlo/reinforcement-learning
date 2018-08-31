import argparse
import random
import numpy as np
import tensorflow as tf


# TODO: pass value_prime


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def flatten_batch_horizon(array):
    b, h, *shape = array.shape
    return array.reshape((b * h, *shape))


# TODO: rename
def batch_a3c_return(rewards, value_prime, dones, gamma):
    batch_size, horizon = rewards.shape
    returns = np.zeros((batch_size, horizon))
    ret = value_prime
    masks = np.logical_not(dones)

    for t in reversed(range(horizon)):
        ret = rewards[:, t] + masks[:, t] * gamma * ret
        returns[:, t] = ret

    return returns


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


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument('--seed', type=int, default=42)
