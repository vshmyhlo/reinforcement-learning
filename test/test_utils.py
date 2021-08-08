import torch

import utils as utils


def test_total_return():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    actual = utils.total_discounted_return(rewards, gamma=0.9)

    expected = torch.tensor(
        [
            [
                1 + 0.9 * 2 + 0.9 ** 2 * 3,
                2 + 0.9 * 3,
                3,
            ]
        ]
    )

    assert torch.allclose(actual, expected)


def test_n_step_discounted_return():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    value_prime = torch.tensor([4.0])
    dones = torch.tensor([[False, True, False]])
    actual = utils.compute_n_step_discounted_return(rewards, value_prime, dones, gamma=0.9)

    expected = torch.tensor(
        [
            [
                2.8,
                2.0,
                6.6,
            ]
        ]
    )

    assert torch.allclose(actual, expected)


def test_n_step_discounted_return_2():
    rewards = torch.tensor([[1.0], [1.0]])
    value_prime = torch.tensor([2.0, 2.0])
    dones = torch.tensor([[False], [True]])
    actual = utils.compute_n_step_discounted_return(rewards, value_prime, dones, gamma=0.9)

    expected = torch.tensor(
        [
            [1.0 + 0.9 * 2.0],
            [1.0],
        ]
    )

    assert torch.allclose(actual, expected)


def test_n_step_discounted_return_3():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    dones = torch.tensor([[False, True, False, False, True]])
    actual = utils.compute_n_step_discounted_return(
        rewards[:, :4], rewards[:, 4], dones[:, :4], gamma=0.9
    )

    expected = torch.cat(
        [
            utils.total_discounted_return(rewards[:, :2], gamma=0.9),
            utils.total_discounted_return(rewards[:, 2:], gamma=0.9)[:, :2],
        ],
        1,
    )

    assert torch.allclose(actual, expected)


def test_generalized_advantage_estimation():
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float)
    values = torch.tensor(
        [
            [3.0, 4.0, 5.0, 3.0, 4.0, 5.0],
        ],
        dtype=torch.float,
    )
    value_prime = torch.tensor([6.0], dtype=torch.float)
    dones = torch.tensor([[False, False, True, False, False, False]], dtype=torch.bool)

    actual = utils.generalized_advantage_estimation(
        rewards, values, value_prime, dones, gamma=0.9, lam=0.8
    )
    expected = torch.tensor([[0.6064, -1.38, -4.0, 3.40576, 2.508, 1.4]])

    assert torch.allclose(actual, expected)
