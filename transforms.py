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


def gridworld(input):
    return input['image'][:, :, 0].astype(np.int64)


def apply_transforms(env, transforms):
    for transform in transforms:
        if transform.type == 'grayscale':
            env = gym.wrappers.GrayScaleObservation(env)
        elif transform.type == 'stack':
            env = wrappers.StackObservation(env, k=transform.k, dim=transform.dim)
        elif transform.type == 'skip':
            env = wrappers.SkipObservation(env, k=transform.k)
        elif transform.type == 'permute':
            env = gym.wrappers.TransformObservation(env, permute)
        elif transform.type == 'normalize':
            env = gym.wrappers.TransformObservation(env, normalize)
        elif transform.type == 'gridworld':
            env = gym.wrappers.TransformObservation(env, gridworld)
        elif transform.type == 'float':
            env = gym.wrappers.TransformObservation(env, lambda s: s.astype(np.float32))
        else:
            raise AssertionError('invalid transform.type {}'.format(transform.type))

    return env
