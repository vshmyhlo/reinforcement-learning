import argparse

import gym
import gym.wrappers
import gym_minigrid
import numpy as np
import torch
import torch.nn as nn
from all_the_tools.metrics import Mean, Last, FPS
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wrappers
import wrappers.torch
from algos.common import build_optimizer
from algos.config import build_default_config
from history import History
from model import Model
from transforms import apply_transforms
from utils import total_discounted_return

gym_minigrid

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: train/eval
# TODO: return normalization


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log/ac-mc')
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--no-render', action='store_true')

    return parser


def build_env(config):
    env = gym.make(config.env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.RescaleAction(env, 0., 1.)
    env = apply_transforms(env, config.transforms)

    return env


def main():
    args = build_parser().parse_args()
    config = build_default_config()
    config.merge_from_file(args.config_path)
    config.experiment_path = args.experiment_path
    config.restore_path = args.restore_path
    config.render = not args.no_render
    config.freeze()
    del args

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
        'ep/reward': Mean(),
        'rollout/entropy': Mean(),
    }

    # ==================================================================================================================
    # training loop
    model.train()
    for episode in tqdm(range(config.episodes), desc='training'):
        history = History()
        s = env.reset()

        with torch.no_grad():
            while True:
                a, _ = model(s)
                a = a.sample()
                s_prime, r, d, meta = env.step(a)
                history.append(state=s, action=a, reward=r)

                if d:
                    break
                else:
                    s = s_prime

        rollout = history.full_rollout()
        dist, values = model(rollout.states)
        returns = total_discounted_return(rollout.rewards, gamma=config.gamma)

        # critic
        errors = returns - values
        critic_loss = errors**2

        # actor
        advantages = errors.detach()
        if isinstance(env.action_space, gym.spaces.Box):
            advantages = advantages.unsqueeze(-1)
        actor_loss = -(dist.log_prob(rollout.actions) * advantages)
        actor_loss -= config.entropy_weight * dist.entropy()
        if isinstance(env.action_space, gym.spaces.Box):
            actor_loss = actor_loss.mean(-1)

        loss = (actor_loss + critic_loss * 0.5).mean(1)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))
        metrics['rollout/entropy'].update(dist.entropy().data.cpu().numpy())

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        metrics['eps'].update(1)
        metrics['ep/length'].update(meta[0]['episode']['l'])
        metrics['ep/reward'].update(meta[0]['episode']['r'])

        if episode % config.log_interval == 0 and episode > 0:
            for k in metrics:
                writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
            writer.add_histogram('rollout/action', rollout.actions, global_step=episode)
            writer.add_histogram('rollout/reward', rollout.rewards, global_step=episode)
            writer.add_histogram('rollout/return', returns, global_step=episode)
            writer.add_histogram('rollout/value', values, global_step=episode)
            writer.add_histogram('rollout/advantage', advantages, global_step=episode)


if __name__ == '__main__':
    main()
