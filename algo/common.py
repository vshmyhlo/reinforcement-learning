import gym
import gym.wrappers
import torch

from transforms import apply_transforms


def build_optimizer(optimizer, parameters):
    if optimizer.type == "momentum":
        return torch.optim.SGD(parameters, optimizer.lr, momentum=0.9, weight_decay=0.0)
    elif optimizer.type == "rmsprop":
        return torch.optim.RMSprop(parameters, optimizer.lr, weight_decay=0.0)
    elif optimizer.type == "adam":
        return torch.optim.Adam(parameters, optimizer.lr, weight_decay=0.0)
    else:
        raise AssertionError("invalid optimizer.type {}".format(optimizer.type))


def build_env(config):
    env = gym.make(config.env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if isinstance(env.action_space, gym.spaces.Box):
        assert env.action_space.is_bounded()
        env = gym.wrappers.RescaleAction(env, 0.0, 1.0)
        # env = gym.wrappers.ClipAction(env)
    env = apply_transforms(env, config.transforms)

    return env
