import numpy as np
import torch
import torch_rl.utils as utils


def test_batch_return():
    rewards = torch.tensor([[1, 2, 3]], dtype=torch.float32)
    actual = utils.total_return(rewards, gamma=0.9)
    expected = [[
        1 + 0.9 * 2 + 0.9**2 * 3,
        2 + 0.9 * 3,
        3
    ]]

    assert np.allclose(actual, expected)


def test_batch_n_step_return():
    rewards = torch.tensor([[1, 2, 3]]).float()
    value_prime = torch.tensor([4]).float()
    dones = torch.tensor([[False, True, False]])
    actual = utils.n_step_return(rewards, value_prime, dones, gamma=0.9)
    expected = [[
        1 + 0.9 * 2,
        2,
        3 + 0.9 * 4
    ]]

    assert np.allclose(actual, expected)
