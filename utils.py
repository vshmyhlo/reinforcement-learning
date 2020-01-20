import argparse

import torch


def total_discounted_return(rewards, gamma):
    value_prime = torch.zeros(rewards.size(0), dtype=rewards.dtype, device=rewards.device)
    dones = torch.full_like(rewards, False, dtype=torch.bool, device=rewards.device)

    return n_step_discounted_return(rewards, value_prime, dones, gamma)


def n_step_discounted_return(rewards, value_prime, dones, gamma):
    assert rewards.dim() == dones.dim() == 2
    assert value_prime.dim() == 1
    assert rewards.size(1) == dones.size(1)

    masks = (~dones).float()
    ret = value_prime
    returns = torch.zeros_like(rewards)

    for t in reversed(range(rewards.size(1))):
        ret = rewards[:, t] + masks[:, t] * gamma * ret
        returns[:, t] = ret

    return returns


# TODO: test
def generalized_advantage_estimation(rewards, values, value_prime, dones, gamma, lam):
    values_prime = torch.cat([values[:, 1:], value_prime.unsqueeze(1)], 1)
    masks = (1 - dones).float()
    gae = torch.zeros(rewards.size(0))
    gaes = torch.zeros_like(rewards)

    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + masks[:, t] * gamma * values_prime[:, t] - values[:, t]
        gae = delta + masks[:, t] * gamma * lam * gae
        gaes[:, t] = gae

    return gaes


def normalize(input):
    return (input - input.mean()) / input.std()


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument('--seed', type=int, default=42)
