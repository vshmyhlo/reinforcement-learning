import argparse
import itertools

import gym
import gym.wrappers
import numpy as np
import torch
from all_the_tools.metrics import Mean, Last, FPS
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wrappers
from algorithms_v1.common import build_optimizer, transform_env
from algorithms_v1.config import build_default_config
from history import History
from model import Model
from utils import total_discounted_return

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: make shared weights work
# TODO: normalize advantage
# TODO: train/eval
# TODO: bn update
# TODO: return normalization
# TODO: monitored session
# TODO: normalize advantage?


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log/pg-mc')
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--no-render', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    config = build_default_config()
    config.merge_from_file(args.config_path)
    config.experiment_path = args.experiment_path
    config.restore_path = args.restore_path
    config.render = not args.no_render
    config.freeze()
    del args

    seed_torch(config.seed)
    env = gym.make(config.env)
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.RescaleAction(env, 0., 1.)
    env = gym.wrappers.TransformObservation(env, transform_env(config.transform))
    env = wrappers.Batch(env)
    if config.render:
        env = wrappers.TensorboardBatchMonitor(env, config.experiment_path, 10)
    env = wrappers.Torch(env, device=DEVICE)
    env.seed(config.seed)
    writer = SummaryWriter(config.experiment_path)

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
        'step/entropy': Mean(),
    }

    # ==================================================================================================================
    # training loop
    model.train()
    for episode in tqdm(range(config.episodes), desc='training'):
        history = History()
        s = env.reset()
        ep_reward = 0

        with torch.no_grad():
            for ep_length in itertools.count():
                a, _ = model(s.float())
                a = a.sample()
                s_prime, r, d, _ = env.step(a)
                ep_reward += r
                history.append(state=s.float(), action=a, reward=r)

                if d:
                    break
                else:
                    s = s_prime

        rollout = history.build_rollout()
        dist, _ = model(rollout.states)
        returns = total_discounted_return(rollout.rewards, gamma=config.gamma)

        # actor
        advantages = returns.detach()
        if isinstance(env.action_space, gym.spaces.Box):
            advantages = advantages.unsqueeze(-1)
        actor_loss = -(dist.log_prob(rollout.actions) * advantages)
        actor_loss -= config.entropy_weight * dist.entropy()
        if isinstance(env.action_space, gym.spaces.Box):
            actor_loss = actor_loss.mean(-1)

        loss = actor_loss.sum(1)

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))
        metrics['eps'].update(1)
        metrics['ep/length'].update(ep_length)
        metrics['ep/reward'].update(ep_reward.data.cpu().numpy())
        metrics['step/entropy'].update(dist.entropy().data.cpu().numpy())

        if episode % config.log_interval == 0 and episode > 0:
            for k in metrics:
                writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
            writer.add_histogram('step/action', rollout.actions, global_step=episode)
            writer.add_histogram('step/reward', rollout.rewards, global_step=episode)
            writer.add_histogram('step/return', returns, global_step=episode)
            writer.add_histogram('step/advantage', advantages, global_step=episode)


if __name__ == '__main__':
    main()