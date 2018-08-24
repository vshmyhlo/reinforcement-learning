import argparse
import random
import numpy as np
import tensorflow as tf


# TODO: test discounted_return

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def discounted_return(history, gamma):
    state, action, reward = zip(*history)
    ret = [sum(gamma**j * r for j, r in enumerate(reward[i:])) for i in range(len(reward))]

    return state, action, ret


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument('--seed', type=int, default=42)
