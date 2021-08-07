from collections import namedtuple

import gym
import gym_minigrid
import torch
import torch.nn as nn
from all_the_tools.config import Config as C
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wrappers
from history import History
from utils import compute_n_step_discounted_return


class Agent(nn.Module):
    def __init__(self, observation_space, action_space: gym.spaces.Discrete):
        super().__init__()
        self.action_value = nn.Linear(observation_space.shape[0], action_space.n)

    def forward(self, obs):
        obs = obs.float()
        action_value = self.action_value(obs)
        return action_value


def main():
    config = C(
        horizon=32,
        discount=0.99,
    )

    writer = SummaryWriter("./log/sarsa_nn")

    # env = gym.make("MiniGrid-Empty-Random-6x6-v0")
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym_minigrid.wrappers.OneHotPartialObsWrapper(env)
    # env = gym_minigrid.wrappers.ImgObsWrapper(env)
    env = wrappers.Batch(env)
    env = wrappers.Torch(env)

    model = Agent(env.observation_space, env.action_space)
    optimizer = torch.optim.RMSprop(model.parameters(), 0.01, alpha=0.99, eps=1e-8)

    obs = env.reset()
    action_value = model(obs)
    action = select_action(action_value)
    episode = 0

    for _ in tqdm(range(10000)):
        history = History()

        for i in range(config.horizon):
            transition = history.append_transition()
            transition.record(action_value_i=select_action_value(action_value, action))

            obs_prime, reward, done, info = env.step(action)
            transition.record(reward=reward, done=done)

            action_value_prime = model(obs_prime)
            action_prime = select_action(action_value_prime)

            action_value, action = action_value_prime, action_prime

            for i in info:
                if "episode" not in i:
                    continue
                episode += 1
                writer.add_scalar("return", info["episode"]["r"], global_step=episode)
                writer.add_scalar("length", info["episode"]["l"], global_step=episode)

        rollout = history.build()

        return_ = compute_n_step_discounted_return(
            reward_t=rollout.reward,
            value_prime=select_action_value(action_value_prime, action_prime).detach(),
            done_t=rollout.done,
            gamma=config.discount,
        )

        td_error = rollout.action_value_i - return_
        loss = td_error.pow(2).mean()

        loss.backward()
        optimizer.step()


def select_action(action_value):
    b, n = action_value.shape
    argmax = action_value.argmax(1)
    mask = torch.rand(size=[b]) < 0.9
    action = torch.where(mask, argmax, torch.randint(0, n, size=[b]))
    return action


def select_action_value(action_value, action):
    b, n = action_value.shape
    return action_value[torch.arange(b), action]


if __name__ == "__main__":
    main()
