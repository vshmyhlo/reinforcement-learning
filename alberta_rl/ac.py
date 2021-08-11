from collections import namedtuple

import gym
import gym_minigrid
import torch
import torch.nn as nn
from all_the_tools.config import Config as C
from tqdm import tqdm

import wrappers
from history import History
from utils import n_step_bootstrapped_return

Output = namedtuple("Output", ["dist", "value"])


class ActorCriticModel(nn.Module):
    def __init__(self, observation_space, action_space: gym.spaces.Discrete):
        super().__init__()

        self.value = nn.Linear(32, 1)
        self.logits = nn.Linear(32, action_space.n)

    def forward(self, obs):
        obs = torch.randn(1, 32)

        logits = self.logits(obs)

        dist = torch.distributions.Categorical(logits=logits)
        value = self.value(obs).squeeze(1)

        return Output(dist=dist, value=value)


def main():
    config = C(
        horizon=32,
        discount=0.99,
    )

    env = gym.make("MiniGrid-Empty-Random-6x6-v0")
    env = gym_minigrid.wrappers.OneHotPartialObsWrapper(env)
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    env = wrappers.Batch(env)
    env = wrappers.Torch(env)

    model = ActorCriticModel(env.observation_space, env.action_space)
    optimizer = torch.optim.RMSprop(model.parameters(), 0.01, alpha=0.99, eps=1e-8)

    obs = env.reset()
    output = model(obs)

    for _ in tqdm(range(10000)):
        history = History()

        for i in range(config.horizon):
            transition = history.append_transition()
            transition.record(value=output.value)

            action = output.dist.sample()
            obs, reward, done, _ = env.step(action)
            transition.record(reward=reward, done=done)

            output_prime = model(obs)
            output = output_prime

        history = history.build()

        return_ = n_step_bootstrapped_return(
            history.reward, output_prime.value.detach(), history.done, config.discount
        )

        td_error = history.value - return_

        loss = td_error.pow(2).mean()

        loss.backward()
        optimizer.step()

        # loss = td_error.mean()
        # value_target =


if __name__ == "__main__":
    main()
