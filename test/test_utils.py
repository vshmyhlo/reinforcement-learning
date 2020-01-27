import torch

import utils as utils


def test_total_return():
    rewards = torch.tensor([[1., 2., 3.]])
    actual = utils.total_discounted_return(rewards, gamma=0.9)

    expected = torch.tensor([[
        1 + 0.9 * 2 + 0.9**2 * 3,
        2 + 0.9 * 3,
        3,
    ]])

    assert torch.allclose(actual, expected)


def test_n_step_discounted_return():
    rewards = torch.tensor([[1., 2., 3.]])
    value_prime = torch.tensor([4.])
    dones = torch.tensor([[False, True, False]])
    actual = utils.n_step_discounted_return(rewards, value_prime, dones, gamma=0.9)

    expected = torch.tensor([[
        1 + 0.9 * 2,
        2,
        3 + 0.9 * 4,
    ]])

    assert torch.allclose(actual, expected)


def test_n_step_discounted_return_2():
    rewards = torch.tensor([[1.], [1.]])
    value_prime = torch.tensor([2., 2.])
    dones = torch.tensor([[False], [True]])
    actual = utils.n_step_discounted_return(rewards, value_prime, dones, gamma=0.9)

    expected = torch.tensor([
        [1. + 0.9 * 2.],
        [1.],
    ])

    assert torch.allclose(actual, expected)


def test_n_step_discounted_return_3():
    rewards = torch.tensor([[1., 2., 3., 4., 5.]])
    dones = torch.tensor([[False, True, False, False, True]])
    actual = utils.n_step_discounted_return(rewards[:, :4], rewards[:, 4], dones[:, :4], gamma=0.9)

    expected = torch.cat([
        utils.total_discounted_return(rewards[:, :2], gamma=0.9),
        utils.total_discounted_return(rewards[:, 2:], gamma=0.9)[:, :2],
    ], 1)

    assert torch.allclose(actual, expected)
