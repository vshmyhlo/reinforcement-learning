import os

import click
import gym
import gym.wrappers
import gym_minigrid
import numpy as np
import pybulletgym
import torch
import torch.nn as nn
from all_the_tools.config import load_config
from all_the_tools.metrics import Mean, Last, FPS
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils
import wrappers
import wrappers.torch
from algo.common import build_optimizer, build_env
from history import History
from model import Model

pybulletgym
gym_minigrid

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: train/eval
# TODO: return normalization


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
    env = wrappers.Batch(build_env(config))
    if config.render:
        env = wrappers.TensorboardBatchMonitor(env, writer, config.log_interval)
    env = wrappers.torch.Torch(env, device=DEVICE)
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
        'rollout/reward': Mean(),
        'rollout/value': Mean(),
        'rollout/advantage': Mean(),
        'rollout/entropy': Mean(),
    }

    # training loop ====================================================================================================
    for episode in tqdm(range(config.episodes), desc='training'):
        hist = History()
        s = env.reset()
        h = model.zero_state(1)
        d = torch.ones(1, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            while True:
                trans = hist.append_transition()

                trans.record(state=s, hidden=h, done=d)
                a, _, h = model(s, h, d)
                a = a.sample()
                s, r, d, info = env.step(a)
                trans.record(action=a, reward=r)

                if d:
                    break

        # optimization =================================================================================================
        model.train()

        # build rollout
        rollout = hist.full_rollout()

        # loss
        loss = compute_loss(env, model, rollout, metrics, config)

        # metrics
        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_last_lr()))

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        metrics['eps'].update(1)
        metrics['ep/length'].update(info[0]['episode']['l'])
        metrics['ep/return'].update(info[0]['episode']['r'])

        if episode % config.log_interval == 0 and episode > 0:
            for k in metrics:
                writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
            torch.save(
                model.state_dict(),
                os.path.join(config.experiment_path, 'model_{}.pth'.format(episode)))


def compute_loss(env, model, rollout, metrics, config):
    dist, values, _ = model(rollout.state, rollout.hidden[:, 0], rollout.done)
    returns = utils.total_discounted_return(rollout.reward, gamma=config.gamma)

    # critic
    errors = returns - values
    critic_loss = errors**2

    # actor
    advantages = errors.detach()
    if config.adv_norm:
        advantages = utils.normalize(advantages)

    log_prob = dist.log_prob(rollout.action)
    entropy = dist.entropy()

    if isinstance(env.action_space, gym.spaces.Box):
        log_prob = log_prob.sum(-1)
        entropy = entropy.sum(-1)

    actor_loss = -log_prob * advantages + \
                 config.entropy_weight * -entropy

    # loss
    loss = (actor_loss + 0.5 * critic_loss).mean(1)

    # metrics
    metrics['rollout/reward'].update(rollout.reward.data.cpu().numpy())
    metrics['rollout/value'].update(values.data.cpu().numpy())
    metrics['rollout/advantage'].update(advantages.data.cpu().numpy())
    metrics['rollout/entropy'].update(entropy.data.cpu().numpy())

    return loss


if __name__ == '__main__':
    main()
