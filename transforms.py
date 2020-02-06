import gym.wrappers
import numba
import numpy as np

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


def gridworld(input):
    return input['image'][:, :, 0].astype(np.int64)


def apply_transforms(env, transforms):
    for transform in transforms:
        if transform.type == 'grayscale':
            env = gym.wrappers.GrayScaleObservation(env)
        elif transform.type == 'stack':
            env = wrappers.StackObservation(env, k=transform.k)
        elif transform.type == 'permute_and_normalize':
            env = gym.wrappers.TransformObservation(env, permute_and_normalize)
        elif transform.type == 'gridworld':
            env = gym.wrappers.TransformObservation(env, gridworld)
        else:
            raise AssertionError('invalid transform.type {}'.format(transform.type))

    return env
