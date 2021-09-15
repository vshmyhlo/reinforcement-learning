import math
import random

import numpy as np
import torch


class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())


class MovingAverage:
    def __init__(self, beta):
        self.beta = beta
        self.value = 0
        self.n = 0

    def update(self, value):
        self.value = self.beta * self.value + (1 - self.beta) * value
        self.n += 1
        return self.value / (1 - self.beta ** self.n)


def shape_matches(*args, dim):
    match = True
    for i in range(len(args)):
        match = match and args[i].size() == args[0].size()
        match = match and args[i].dim() == dim

    return match


def total_discounted_return(reward_t, gamma: float):
    assert shape_matches(reward_t, dim=2)

    return_ = 0
    return_t = torch.zeros_like(reward_t)

    for t in reversed(range(reward_t.size(1))):
        return_ = reward_t[:, t] + gamma * return_
        return_t[:, t] = return_

    return return_t


def n_step_bootstrapped_return(
    reward_t,
    done_t,
    value_prime,
    discount,
):
    assert shape_matches(reward_t, done_t, dim=2)
    assert shape_matches(value_prime, dim=1)

    mask_t = (~done_t).float()
    return_ = value_prime
    return_t = torch.zeros_like(reward_t)

    for t in reversed(range(reward_t.size(1))):
        return_ = reward_t[:, t] + mask_t[:, t] * discount * return_
        return_t[:, t] = return_

    return return_t


# TODO: test
def one_step_discounted_return(rewards, values_prime, dones, gamma):
    masks = (~dones).float()
    returns = rewards + masks * gamma * values_prime

    return returns


# TODO: detach?
def generalized_advantage_estimation(reward_t, value_t, value_prime, done_t, gamma, lambda_):
    mask_t = (~done_t).float()
    values_prime = torch.cat([value_t[:, 1:], value_prime.unsqueeze(1)], 1)
    td_error = reward_t + mask_t * gamma * values_prime - value_t

    gae = torch.zeros(reward_t.size(0))
    gae_t = torch.zeros_like(reward_t)
    for t in reversed(range(reward_t.size(1))):
        gae = td_error[:, t] + mask_t[:, t] * gamma * lambda_ * gae
        gae_t[:, t] = gae

    return gae_t


def n_step_lambda_bootstrapped_return(
    reward_t,
    value_t,
    value_prime,
    done_t,
    gamma,
    lambda_,
):
    mask_t = (~done_t).float()
    value_prime_t = torch.cat([value_t[:, 1:], value_prime.unsqueeze(1)], 1)

    # return_ = value_prime
    # return_t = torch.zeros_like(reward_t)
    return_lambda_t = torch.zeros_like(reward_t)
    two_step = value_prime
    for t in reversed(range(reward_t.size(1))):
        one_step = reward_t[:, t] + gamma * value_prime_t[:, t] * mask_t[:, t]
        return_lambda = one_step + lambda_ * two_step

        return_lambda_t[:, t] = (1 - lambda_) * return_lambda

        two_step = one_step

        # return_ = reward_t[:, t] + mask_t[:, t] * gamma * return_
        # return_t[:, t] = return_

    # return_lambda = 0
    # return_lambda_t = torch.zeros_like(reward_t)
    # for t in reversed(range(reward_t.size(1))):
    #     return_lambda = return_t[:, t] + mask_t[:, t] * lambda_ * return_lambda
    #     return_lambda_t[:, t] = (1 - lambda_) * return_lambda

    return return_lambda_t


# def generalized_advantage_estimation(rewards, values, value_prime, dones, gamma, lam):
#     masks = (~dones).float()
#     values_prime = torch.cat([values[:, 1:], value_prime.unsqueeze(1)], 1)
#     td_error = rewards + masks * gamma * values_prime - values
#     gaes = torch.zeros_like(rewards)
#
#     for t in range(rewards.size(1)):
#         gae = torch.zeros(rewards.size(0))
#         for l in range(rewards.size(1) - t):
#             gae += (gamma * lam)**l * td_error[:, t + l]
#             if dones[:, t + l]:
#                 break
#         gaes[:, t] = gae
#
#     return gaes


def normalize(input):
    return (input - input.mean()) / input.std()


def random_seed(seed: int):
    # python
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# TODO: test
def differential_n_step_bootstrapped_return(
    reward_t,
    done_t,
    value_prime,
    average_reward,
):
    mask_t = (~done_t).float()
    return_ = value_prime
    return_t = torch.zeros_like(reward_t)

    for t in reversed(range(reward_t.size(1))):
        return_ = reward_t[:, t] - average_reward + mask_t[:, t] * return_
        return_t[:, t] = return_

    return return_t
