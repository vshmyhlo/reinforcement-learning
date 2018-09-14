import torch


# TODO: test

def batch_return(rewards, gamma):
    value_prime = torch.zeros(rewards.shape[:1])
    dones = torch.full(rewards.shape, False)

    return batch_n_step_return(rewards, value_prime, dones, gamma)


def batch_n_step_return(rewards, value_prime, dones, gamma):
    assert rewards.dim() == 2
    assert value_prime.dim() == 1
    assert dones.dim() == 2

    mask = (1 - dones).float()
    ret = value_prime
    returns = torch.zeros_like(rewards)

    for t in reversed(range(rewards.size(1))):
        ret = rewards[:, t] + mask[:, t] * gamma * ret
        returns[:, t] = ret

    return returns
