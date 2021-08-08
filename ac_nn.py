from collections import namedtuple

import click
import gym
import gym_minigrid
import torch
import torch.nn as nn
from all_the_tools.config import Config as C
from all_the_tools.meters import Mean
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wrappers
from history import History
from utils import compute_n_step_discounted_return
from vec_env import VecEnv

torch.autograd.set_detect_anomaly(True)


class Agent(nn.Module):
    def __init__(self, observation_space, action_space: gym.spaces.Discrete):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(20, 16, (2, 2)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 2)),
            #
            nn.Conv2d(16, 32, (2, 2)),
            nn.LeakyReLU(0.2),
            #
            nn.Conv2d(32, 64, (2, 2)),
            nn.LeakyReLU(0.2),
        )
        self.rnn = nn.LSTMCell(64, 64)
        self.dist = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n),
        )
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.apply(self.weight_init)

    def forward(self, input, state):
        input = input.float().permute(0, 3, 1, 2)
        input = self.embedding(input)
        input = input.view(input.size(0), input.size(1))

        state = self.rnn(input, state)
        input, _ = state

        dist = torch.distributions.Categorical(logits=self.dist(input))
        value = self.value(input).squeeze(1)

        return dist, value, state

    def zero_state(self, batch_size):
        zeros = torch.zeros(batch_size, 64)
        state = (zeros, zeros)
        return state

    def reset_state(self, state, done):
        done = done.unsqueeze(1)
        state = tuple(torch.where(done, torch.zeros_like(x), x) for x in state)
        return state

    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


@click.command()
@click.option("--experiment-path", type=click.Path(), required=True)
def main(**kwargs):
    config = C(
        horizon=32,
        discount=0.99,
        num_episodes=100000,
        num_workers=8,
        entropy_weight=1e-2,
    )
    for k in kwargs:
        config[k] = kwargs[k]

    writer = SummaryWriter(config.experiment_path)

    env = VecEnv([build_env for _ in range(config.num_workers)])
    env = wrappers.TensorboardBatchMonitor(env, writer, log_interval=100, fps_mul=0.5)
    env = wrappers.Torch(env)

    model = Agent(env.observation_space, env.action_space)
    optimizer = torch.optim.RMSprop(model.parameters(), 1e-4 * config.num_workers)

    episode = 0
    pbar = tqdm(total=config.num_episodes)

    obs = env.reset()
    state = model.zero_state(config.num_workers)

    while episode < config.num_episodes:
        history = History()
        state = tuple(x.detach() for x in state)

        for i in range(config.horizon):
            transition = history.append_transition()

            dist, value, state_prime = model(obs, state)
            transition.record(value=value, entropy=dist.entropy())
            action = select_action(dist)
            transition.record(action_log_prob=dist.log_prob(action))

            obs_prime, reward, done, info = env.step(action)
            transition.record(reward=reward, done=done)
            state_prime = model.reset_state(state_prime, done)

            obs, state = obs_prime, state_prime

            for i in info:
                if "episode" not in i:
                    continue
                episode += 1
                writer.add_scalar("episode/return", i["episode"]["r"], global_step=episode)
                writer.add_scalar("episode/length", i["episode"]["l"], global_step=episode)
                pbar.update()

        rollout = history.build()

        _, value_prime, _ = model(obs_prime, state_prime)

        return_ = compute_n_step_discounted_return(
            reward_t=rollout.reward,
            value_prime=value_prime.detach(),
            done_t=rollout.done,
            gamma=config.discount,
        )

        td_error = rollout.value - return_
        critic_loss = td_error.pow(2)
        actor_loss = (
            -rollout.action_log_prob * td_error.detach() - config.entropy_weight * rollout.entropy
        )
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        writer.add_scalar("rollout/value", rollout.value.mean(), global_step=episode)
        writer.add_scalar("rollout/td_error", td_error.mean(), global_step=episode)
        writer.add_scalar("rollout/loss", loss.mean(), global_step=episode)
        writer.add_scalar("rollout/actor_loss", actor_loss.mean(), global_step=episode)
        writer.add_scalar("rollout/critic_loss", critic_loss.mean(), global_step=episode)
        writer.add_scalar("rollout/entropy", rollout.entropy.mean(), global_step=episode)

    env.close()
    writer.close()


def build_env():
    # env = gym.make("CartPole-v1")
    env = "MiniGrid-Empty-Random-6x6-v0"
    env = gym.make(env)
    env = gym_minigrid.wrappers.OneHotPartialObsWrapper(env)
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def select_action(dist: torch.distributions.Distribution):
    return dist.sample()


if __name__ == "__main__":
    main()
