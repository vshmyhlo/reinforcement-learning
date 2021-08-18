import torch

import utils


def test_moving_average():
    ma = utils.MovingAverage(0.9)
    assert ma.update(10) == 10
    assert ma.update(10) == 10
    assert ma.update(10) == 10


def test_total_discounted_return():
    reward_t = torch.tensor([[1.0, 2.0, 3.0]])
    actual = utils.total_discounted_return(reward_t, gamma=0.9)
    expected = torch.tensor([[5.23, 4.7, 3]])

    assert torch.allclose(actual, expected)


def test_n_step_bootstrapped_return():
    reward_t = torch.tensor([[1.0, 2.0, 3.0]])
    done_t = torch.tensor([[False, False, True]])
    value_prime = torch.tensor([4.0])

    actual = utils.n_step_bootstrapped_return(reward_t, done_t, value_prime, discount=0.9)
    expected = torch.tensor([[5.23, 4.7, 3]])

    assert torch.allclose(actual, expected)


def test_n_step_bootstrapped_return_2():
    reward_t = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]])
    done_t = torch.tensor([[False, False, True, False, False, False]])
    value_prime = torch.tensor([4.0])

    actual = utils.n_step_bootstrapped_return(reward_t, done_t, value_prime, discount=0.9)
    expected = torch.tensor([[5.23, 4.7, 3, 8.146, 7.94, 6.6]])

    assert torch.allclose(actual, expected)


def test_generalized_advantage_estimation():
    reward_t = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float)
    value_t = torch.tensor([[3.0, 4.0, 5.0, 3.0, 4.0, 5.0]], dtype=torch.float)
    value_prime = torch.tensor([6.0], dtype=torch.float)
    done_t = torch.tensor([[False, False, True, False, False, False]], dtype=torch.bool)

    actual = utils.generalized_advantage_estimation(
        reward_t, value_t, value_prime, done_t, gamma=0.9, lambda_=0.8
    )
    expected = torch.tensor([[0.6064, -1.38, -4.0, 3.40576, 2.508, 1.4]])

    assert torch.allclose(actual, expected)


def gae_ref(reward_t, value_t, value_prime, done_t, gamma, lambda_):
    mb_advs = torch.zeros_like(reward_t)
    lastgaelam = 0

    for t in reversed(range(reward_t.size(1))):
        if t == reward_t.size(1) - 1:
            nextnonterminal = 1.0 - done_t[:, t].float()
            nextvalues = value_prime
        else:
            nextnonterminal = 1.0 - done_t[:, t].float()
            nextvalues = value_t[:, t + 1]
        delta = reward_t[:, t] + gamma * nextvalues * nextnonterminal - value_t[:, t]
        mb_advs[:, t] = lastgaelam = delta + gamma * lambda_ * nextnonterminal * lastgaelam

    return mb_advs


def test_generalized_advantage_estimation_ref():
    reward_t = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float)
    value_t = torch.tensor([[3.0, 4.0, 5.0, 3.0, 4.0, 5.0]], dtype=torch.float)
    value_prime = torch.tensor([6.0], dtype=torch.float)
    done_t = torch.tensor([[False, False, True, False, False, False]], dtype=torch.bool)

    actual = utils.generalized_advantage_estimation(
        reward_t, value_t, value_prime, done_t, gamma=0.9, lambda_=0.8
    )
    expected = gae_ref(reward_t, value_t, value_prime, done_t, gamma=0.9, lambda_=0.8)

    assert torch.allclose(actual, expected)


def gae_ref_2(reward_t, value_t, value_prime, done_t, gamma, lambda_):
    advantages = torch.zeros_like(reward_t)
    mask_t = (~done_t).float()

    for t in reversed(range(reward_t.size(1))):
        next_mask = mask_t[:, t]
        next_value = value_t[:, t + 1] if t < reward_t.size(1) - 1 else value_prime
        next_advantage = advantages[:, t + 1] if t < reward_t.size(1) - 1 else 0

        delta = reward_t[:, t] + gamma * next_value * next_mask - value_t[:, t]
        advantages[:, t] = delta + gamma * lambda_ * next_advantage * next_mask

    return advantages


def test_generalized_advantage_estimation_ref_2():
    reward_t = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float)
    value_t = torch.tensor([[3.0, 4.0, 5.0, 3.0, 4.0, 5.0]], dtype=torch.float)
    value_prime = torch.tensor([6.0], dtype=torch.float)
    done_t = torch.tensor([[False, False, True, False, False, False]], dtype=torch.bool)

    actual = utils.generalized_advantage_estimation(
        reward_t, value_t, value_prime, done_t, gamma=0.9, lambda_=0.8
    )
    expected = gae_ref_2(reward_t, value_t, value_prime, done_t, gamma=0.9, lambda_=0.8)

    assert torch.allclose(actual, expected)


#
#
# # def test_tmp_2():
# #     reward_t = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float)
# #     value_t = torch.tensor([[3.0, 4.0, 5.0, 3.0, 4.0, 5.0]], dtype=torch.float)
# #     value_prime = torch.tensor([6.0], dtype=torch.float)
# #     done_t = torch.tensor([[False, False, True, False, False, False]], dtype=torch.bool)
# #
# #     lambda_return = utils.compute_n_step_lambda_bootstrapped_return(
# #         reward_t, value_t, value_prime, done_t, gamma=0.9, lambda_=0.0
# #     )
# #     expected = torch.tensor([[4.6, 5.5, 1.0, 4.6, 5.5, 6.4]], dtype=torch.float)
# #
# #     assert torch.allclose(lambda_return, expected)
# #
# #
# # def test_tmp_3():
# #     reward_t = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float)
# #     value_t = torch.tensor([[3.0, 4.0, 5.0, 3.0, 4.0, 5.0]], dtype=torch.float)
# #     value_prime = torch.tensor([6.0], dtype=torch.float)
# #     done_t = torch.tensor([[False, False, True, False, False, False]], dtype=torch.bool)
# #
# #     lambda_return = utils.compute_n_step_lambda_bootstrapped_return(
# #         reward_t, value_t, value_prime, done_t, gamma=0.9, lambda_=1.0
# #     )
# #     expected = torch.tensor([[12.0, 5.5, 1.0, 4.6, 5.5, 6.4]], dtype=torch.float)
# #
# #     assert torch.allclose(lambda_return, expected)
# #
# #
