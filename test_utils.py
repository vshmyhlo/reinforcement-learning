import numpy as np
import utils


# def test_discounted_return():
#     rewards = [[-1, 0, 1]]
#     actual = utils.discounted_return(rewards, 0.9)
#
#     expected = [[
#         5 + 0.9 * 4 + 0.9**2 * 3,
#         4 + 0.9 * 3,
#         3
#     ]]
#
#     assert np.array_equal(actual, expected)

def test_discounted_reward():
    rewards = np.array([[5, 4, 3]])
    actual = utils.discounted_reward(rewards, 0.9)
    expected = np.array([5 + 0.9 * 4 + 0.9**2 * 3])

    assert np.allclose(actual, expected)


# def test_discounted_return():
#     rewards = np.array([[5, 4, 3]])
#     actual = utils.discounted_return(rewards, 0.9)
#     expected = np.array([[
#         5 + 0.9 * 4 + 0.9**2 * 3,
#         4 + 0.9 * 3,
#         3
#     ]])
#
#     assert np.allclose(actual, expected)


def test_batch_a3c_return():
    rewards = np.array([[5, 4, 3] * 2 + [2]])
    value_prime = np.array([100])
    dones = np.array([[False, False, True] * 2 + [False]])

    actual = utils.batch_a3c_return(rewards, value_prime, dones, gamma=0.9)
    expected = [[
                    5 + 0.9 * 4 + 0.9**2 * 3,
                    4 + 0.9 * 3,
                    3
                ] * 2 + [92]]

    assert np.allclose(actual, expected)


def test_generalized_advantage_estimation():
    rewards = np.array([[-1, 0, 1] * 2])
    values = np.array([[5, 4, 3] * 2])
    value_prime = np.array([100])
    dones = np.array([[False, False, True] * 2])

    actual = utils.generalized_advantage_estimation(rewards, values, value_prime, dones, gamma=0.9, lam=0.8)
    expected = [[-4.3728, -2.74, -2.] * 2]

    assert np.allclose(actual, expected)
