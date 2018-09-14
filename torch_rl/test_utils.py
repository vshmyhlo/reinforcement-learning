import numpy as np
import torch
import torch_rl.utils as utils


def test_batch_return():
    rewards = torch.tensor([[1, 2, 3]], dtype=torch.float32)
    actual = utils.batch_return(rewards, gamma=0.9)
    expected = [[
        1 + 0.9 * 2 + 0.9**2 * 3,
        2 + 0.9 * 3,
        3
    ]]

    assert np.allclose(actual, expected)
