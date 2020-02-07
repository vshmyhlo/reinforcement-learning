import argparse
import os

import gym
import gym.wrappers
import gym_minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from all_the_tools.metrics import Mean, FPS, Last
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wrappers
from history import History
from transforms import apply_transforms
from utils import n_step_discounted_return
from v1.common import build_optimizer
from v1.config import build_default_config
from v1.model import Model
from vec_env import VecEnv

gym_minigrid

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: revisit stat calculation
# TODO: normalize advantage?
# TODO: normalize input (especially images)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log/pg-mc')
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--no-render', action='store_true')

    return parser


# TODO: move to shared code
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
    env = VecEnv([
        lambda: build_env(config)
        for _ in range(config.workers)])
    if config.render:
        env = wrappers.TensorboardBatchMonitor(env, writer, config.log_interval)
    env = wrappers.Torch(env, device=DEVICE)
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
    episode = 0
    s = env.reset()

    bar = tqdm(total=config.episodes, desc='training')
    while episode < config.episodes:
        history = History()

        with torch.no_grad():
            for _ in range(config.horizon):
                a, _ = model(s)
                a = a.sample()
                s_prime, r, d, meta = env.step(a)
                history.append(state=s, action=a, reward=r, done=d)
                s = s_prime

                indices, = torch.where(d)
                for i in indices:
                    metrics['eps'].update(1)
                    metrics['ep/length'].update(meta[i]['episode']['l'])
                    metrics['ep/reward'].update(meta[i]['episode']['r'])
                    episode += 1
                    scheduler.step()
                    bar.update(1)

                    if episode % config.log_interval == 0 and episode > 0:
                        for k in metrics:
                            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
                        writer.add_histogram('rollout/action', rollout.actions, global_step=episode)
                        writer.add_histogram('rollout/reward', rollout.rewards, global_step=episode)
                        writer.add_histogram('rollout/return', returns, global_step=episode)
                        writer.add_histogram('rollout/value', values, global_step=episode)
                        writer.add_histogram('rollout/advantage', advantages, global_step=episode)

                        torch.save(
                            model.state_dict(),
                            os.path.join(config.experiment_path, 'model_{}.pth'.format(episode)))

        rollout = history.build_rollout(s_prime)
        dist, values = model(rollout.states)
        _, value_prime = model(rollout.state_prime)
        value_prime = value_prime.detach()
        returns = n_step_discounted_return(rollout.rewards, value_prime, rollout.dones, gamma=config.gamma)

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

    bar.close()
    env.close()


if __name__ == '__main__':
    main()
