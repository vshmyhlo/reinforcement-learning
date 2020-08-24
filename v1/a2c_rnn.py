import os

import click
import gym
import gym.wrappers
import gym_minigrid
import numpy as np
import pybulletgym
import torch
import torch.nn as nn
import torch.optim
from all_the_tools.config import load_config
from all_the_tools.metrics import Mean, Last, FPS
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wrappers
from history_v2 import History
from transforms import apply_transforms
from utils import n_step_discounted_return
from v1.common import build_optimizer
from v1.model import Model
from vec_env import VecEnv

pybulletgym
gym_minigrid

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: check how finished episodes count
# TODO: revisit stat calculation
# TODO: normalize advantage?
# TODO: how to agregate entropy for each actino
# TODO: normalize input (especially images)
# TODO: refactor EPS (noisy and incorrect statistics)
# TODO: sum or average entropy of each action


# TODO: move to shared code
def build_env(config):
    env = gym.make(config.env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if isinstance(env.action_space, gym.spaces.Box):
        assert env.action_space.is_bounded()
        env = gym.wrappers.RescaleAction(env, 0., 1.)
    env = apply_transforms(env, config.transforms)

    return env


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--restore-path', type=click.Path())
@click.option('--render', is_flag=True)
def main(config_path, **kwargs):
    config = load_config(
        config_path,
        **kwargs)
    del config_path, kwargs

    writer = SummaryWriter(config.experiment_path)

    seed_torch(config.seed)
    env = VecEnv([
        lambda: build_env(config)
        for _ in range(config.workers)])
    if config.render:
        env = wrappers.TensorboardBatchMonitor(env, writer, config.log_interval)
    env = wrappers.Torch(env, dtype=torch.float, device=DEVICE)
    env.seed(config.seed)

    model = Model(config.model, env.observation_space, env.action_space)
    model = model.to(DEVICE)
    if config.restore_path is not None:
        model.load_state_dict(torch.load(config.restore_path))
    optimizer = build_optimizer(config.opt, model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.episodes)

    metrics = {
        'loss': Mean(),
        'lr': Last(),
        'eps': FPS(),
        'ep/length': Mean(),
        'ep/return': Mean(),
        'rollout/entropy': Mean(),
        'rollout/reward': Mean(),
        'rollout/value': Mean(),
        'rollout/advantage': Mean(),
    }

    # ==================================================================================================================
    # training loop
    model.train()
    episode = 0

    s = env.reset()
    h = model.zero_state(config.workers)
    d = torch.ones(config.workers, dtype=torch.bool)

    bar = tqdm(total=config.episodes, desc='training')
    while episode < config.episodes:
        hist = History(['state', 'hidden', 'done'])
        hist_prime = History(['state', 'hidden', 'done', 'action', 'reward'])

        with torch.no_grad():
            for _ in range(config.horizon):
                hist.append(state=s, hidden=h, done=d)
                a, _, h = model(s, h, d)
                a = a.sample()
                s, r, d, info = env.step(a)
                hist_prime.append(state=s, hidden=h, done=d, action=a, reward=r)

                indices, = torch.where(d)
                for i in indices:
                    metrics['eps'].update(1)
                    metrics['ep/length'].update(info[i]['episode']['l'])
                    metrics['ep/return'].update(info[i]['episode']['r'])
                    episode += 1
                    scheduler.step()
                    bar.update(1)

                    if episode % config.log_interval == 0 and episode > 0:
                        for k in metrics:
                            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
                        torch.save(
                            model.state_dict(),
                            os.path.join(config.experiment_path, 'model_{}.pth'.format(episode)))

        # build rollout
        rollout = hist.build_rollout()
        rollout_prime = hist_prime.build_rollout()

        # loss
        loss = compute_loss(env, model, rollout, rollout_prime, metrics, config)

        # metrics
        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    bar.close()
    env.close()


def compute_loss(env, model, rollout, rollout_prime, metrics, config):
    dist, values, hidden = model(rollout.state, rollout.hidden[:, 0], rollout.done)
    with torch.no_grad():
        _, value_prime, _ = model(rollout_prime.state[:, -1], hidden, rollout_prime.done[:, -1])
        returns = n_step_discounted_return(rollout_prime.reward, value_prime, rollout_prime.done, gamma=config.gamma)

    # critic
    errors = returns - values
    critic_loss = errors**2

    # actor
    advantages = errors.detach()
    log_prob = dist.log_prob(rollout_prime.action)
    entropy = dist.entropy()

    if isinstance(env.action_space, gym.spaces.Box):
        log_prob = log_prob.sum(-1)
        entropy = entropy.sum(-1)

    actor_loss = -log_prob * advantages - \
                 config.entropy_weight * entropy

    # loss
    loss = (actor_loss + 0.5 * critic_loss).mean(1)

    # metrics
    metrics['rollout/reward'].update(rollout_prime.reward.data.cpu().numpy())
    metrics['rollout/value'].update(values.data.cpu().numpy())
    metrics['rollout/advantage'].update(advantages.data.cpu().numpy())
    metrics['rollout/entropy'].update(entropy.data.cpu().numpy())

    return loss


if __name__ == '__main__':
    main()
