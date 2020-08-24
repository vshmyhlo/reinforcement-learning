from functools import partial

import cv2
import gym.wrappers
import numba
import numpy as np

import wrappers


def permute(input):
    input = np.moveaxis(input, 2, 0)

    return input


@numba.njit()
def normalize(input):
    input = input.astype(np.float32)
    input -= 255 / 2
    input /= 255 / 2

    return input


def resize(input, size):
    if isinstance(size, int):
        size = (size, size)
    input = cv2.resize(input, size)

    return input


def apply_transforms(env, transforms):
    for transform in transforms:
        if transform.type == 'adj_max':
            env = wrappers.AdjMax(env)
        elif transform.type == 'grayscale':
            env = gym.wrappers.GrayScaleObservation(env)
        elif transform.type == 'resize':
            env = gym.wrappers.TransformObservation(env, partial(resize, size=transform.size))
        elif transform.type == 'stack':
            env = wrappers.StackObs(env, k=transform.k, dim=transform.dim)
        elif transform.type == 'skip':
            env = wrappers.SkipObs(env, k=transform.k)
        elif transform.type == 'permute':
            env = gym.wrappers.TransformObservation(env, permute)
        elif transform.type == 'normalize':
            env = gym.wrappers.TransformObservation(env, normalize)
        elif transform.type == 'gridworld':
            env = wrappers.GridWorld(env)
        else:
            raise AssertionError('invalid transform.type {}'.format(transform.type))

    return env
