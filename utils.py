import torch


def total_discounted_return(rewards, gamma):
    value_prime = torch.zeros(
        rewards.size(0), dtype=rewards.dtype, device=rewards.device
    )
    dones = torch.full_like(rewards, False, dtype=torch.bool, device=rewards.device)
    dones[:, rewards.size(1) - 1] = True

    return compute_n_step_discounted_return(rewards, value_prime, dones, gamma)


def compute_n_step_discounted_return(reward_t, value_prime, done_t, gamma):
    assert reward_t.dim() == done_t.dim() == 2
    assert value_prime.dim() == 1
    assert reward_t.size(1) == done_t.size(1)

    masks = (~done_t).float()
    ret = value_prime
    returns = torch.zeros_like(reward_t)

    for t in reversed(range(reward_t.size(1))):
        ret = reward_t[:, t] + masks[:, t] * gamma * ret
        returns[:, t] = ret

    return returns


# TODO: test
def one_step_discounted_return(rewards, values_prime, dones, gamma):
    masks = (~dones).float()
    returns = rewards + masks * gamma * values_prime

    return returns


# TODO: detach?
def generalized_advantage_estimation(rewards, values, value_prime, dones, gamma, lam):
    masks = (~dones).float()
    values_prime = torch.cat([values[:, 1:], value_prime.unsqueeze(1)], 1)
    td_error = rewards + masks * gamma * values_prime - values
    gaes = torch.zeros_like(rewards)

    gae = torch.zeros(rewards.size(0))
    for t in reversed(range(rewards.size(1))):
        gae = td_error[:, t] + masks[:, t] * gamma * lam * gae
        gaes[:, t] = gae

    return gaes


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
