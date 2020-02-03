import gym.wrappers
import numba
import numpy as np
import torch

import wrappers


def permute_and_normalize(input):
    input = np.moveaxis(input, 2, 0)
    input = normalize(input)

    return input


@numba.njit()
def normalize(input):
    input = input.astype(np.float32)
    input -= 255 / 2
    input /= 255 / 2

    return input


def build_optimizer(optimizer, parameters):
    if optimizer.type == 'momentum':
        return torch.optim.SGD(parameters, optimizer.lr, momentum=0.9, weight_decay=0.)
    elif optimizer.type == 'rmsprop':
        return torch.optim.RMSprop(parameters, optimizer.lr, weight_decay=0.)
    elif optimizer.type == 'adam':
        return torch.optim.Adam(parameters, optimizer.lr, weight_decay=0.)
    else:
        raise AssertionError('invalid optimizer.type {}'.format(optimizer.type))


def transform_env(env, transforms):
    for transform in transforms:
        if transform.type == 'grayscale':
            env = gym.wrappers.GrayScaleObservation(env)
        elif transform.type == 'stack':
            env = wrappers.StackObservation(env, k=transform.k)
        elif transform.type == 'permute_and_normalize':
            env = gym.wrappers.TransformObservation(env, permute_and_normalize)
        else:
            raise AssertionError('invalid transform.type {}'.format(transform.type))

    return env
