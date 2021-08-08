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
        self.action_value = nn.Linear(64, action_space.n)

        self.apply(self.weight_init)

    def forward(self, input, state):
        input = input.float().permute(0, 3, 1, 2)
        input = self.embedding(input)
        input = input.view(input.size(0), input.size(1))

        state = self.rnn(input, state)
        input, _ = state

        action_value = self.action_value(input)

        return action_value, state

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
        e_greedy_eps=0.9,
    )
    for k in kwargs:
        config[k] = kwargs[k]

    writer = SummaryWriter(config.experiment_path)

    env = VecEnv([build_env for _ in range(config.num_workers)])
    env = wrappers.TensorboardBatchMonitor(env, writer, log_interval=100, fps_mul=0.5)
    env = wrappers.Torch(env)

    model = Agent(env.observation_space, env.action_space)
    optimizer = torch.optim.RMSprop(model.parameters(), 1e-4 * config.num_workers)
    # optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # metrics = {
    #     "action_value_i": Mean(),
    # }
    episode = 0
    pbar = tqdm(total=config.num_episodes)

    obs = env.reset()
    state = model.zero_state(config.num_workers)

    while episode < config.num_episodes:
        history = History()
        state = tuple(x.detach() for x in state)

        for i in range(config.horizon):
            transition = history.append_transition()

            action_value, state_prime = model(obs, state)
            action = select_action(action_value, eps=config.e_greedy_eps)
            transition.record(action_value_i=select_action_value(action_value, action))

            obs_prime, reward, done, info = env.step(action)
            state_prime = model.reset_state(state_prime, done)
            transition.record(reward=reward, done=done)

            obs, state = obs_prime, state_prime

            for i in info:
                if "episode" not in i:
                    continue
                episode += 1
                writer.add_scalar("return", i["episode"]["r"], global_step=episode)
                writer.add_scalar("length", i["episode"]["l"], global_step=episode)
                pbar.update()

        rollout = history.build()

        action_value_prime, _ = model(obs_prime, state_prime)
        action_prime = select_action(action_value_prime, eps=config.e_greedy_eps)

        return_ = compute_n_step_discounted_return(
            reward_t=rollout.reward,
            value_prime=select_action_value(action_value_prime, action_prime).detach(),
            done_t=rollout.done,
            gamma=config.discount,
        )

        td_error = rollout.action_value_i - return_
        loss = td_error.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar(
            "rollout/action_value_i", rollout.action_value_i.mean(), global_step=episode
        )
        writer.add_scalar("rollout/td_error", td_error.mean(), global_step=episode)
        writer.add_scalar("rollout/loss", loss.mean(), global_step=episode)

    env.close()
    writer.close()


def build_env():
    # env = gym.make("CartPole-v1")

    # env = gym.make("MiniGrid-Empty-Random-6x6-v0")
    env = gym.make("MiniGrid-FourRooms-v0")
    env = gym_minigrid.wrappers.OneHotPartialObsWrapper(env)
    env = gym_minigrid.wrappers.ImgObsWrapper(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def select_action(action_value, eps):
    b, n = action_value.shape
    argmax = action_value.argmax(1)
    mask = torch.rand(size=[b]) < eps
    action = torch.where(mask, argmax, torch.randint(0, n, size=[b]))
    return action


def select_action_value(action_value, action):
    b, n = action_value.shape
    return action_value[torch.arange(b), action]


if __name__ == "__main__":
    main()
