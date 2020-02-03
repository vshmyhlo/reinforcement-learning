import argparse
import os

import gym
import gym.wrappers
import numpy as np
import torch
import torch.optim
from all_the_tools.metrics import Mean, FPS, Last
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wrappers
from algorithms_v2.common import build_optimizer, build_transform
from algorithms_v2.config import build_default_config
from algorithms_v2.observable import Observable
from algorithms_v2.rollout_callback import FiniteHorizonRolloutCallback, FullEpisodeRolloutCallback
from model import Model
from utils import n_step_discounted_return
from vec_env import VecEnv

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: torch wrapper
# TODO: revisit stat calculation
# TODO: normalize advantage?
# TODO: normalize input (especially images)


class ActorCriticOptimizationCallback(object):
    def __init__(self, model, writer, config):
        self.model = model
        self.writer = writer
        self.config = config

        self.optimizer = build_optimizer(config.opt, model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.steps / config. )

        self.scalar_metrics = {
            'loss': Mean(),
            'lr': Last(),
            'rollout/entropy': Mean(),
        }

    def on_rollout_done(self, rollout, step):
        dist, values = self.model(rollout.states)
        _, value_prime = self.model(rollout.state_prime)
        value_prime = value_prime.detach()
        returns = n_step_discounted_return(rollout.rewards, value_prime, rollout.dones, gamma=self.config.gamma)

        # critic
        errors = returns - values
        critic_loss = errors**2

        # actor
        advantages = errors.detach()
        actor_loss = -(dist.log_prob(rollout.actions) * advantages)
        actor_loss -= self.config.entropy_weight * dist.entropy()

        loss = (actor_loss + critic_loss).sum(1)

        self.scalar_metrics['loss'].update(loss.data.cpu().numpy())
        self.scalar_metrics['lr'].update(np.squeeze(self.scheduler.get_lr()))
        self.scalar_metrics['rollout/entropy'].update(dist.entropy().data.cpu().numpy())

        # training
        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()
        self.scheduler.step()

        for _ in range(self.config.workers):
            step += 1
            if step % self.config.log_interval == 0 and step > 0:
                for k in self.scalar_metrics:
                    self.writer.add_scalar(k, self.scalar_metrics[k].compute_and_reset(), global_step=step)

                self.writer.add_histogram('rollout/action', rollout.actions, global_step=step)
                self.writer.add_histogram('rollout/reward', rollout.rewards, global_step=step)
                self.writer.add_histogram('rollout/return', returns, global_step=step)
                self.writer.add_histogram('rollout/value', values, global_step=step)
                self.writer.add_histogram('rollout/advantage', advantages, global_step=step)

                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.config.experiment_path, 'model_{}.pth'.format(step)))


class Runner(Observable):
    def __init__(self, env, model, config, writer, callbacks):
        super().__init__(callbacks)

        self.env = env
        self.model = model
        self.config = config
        self.writer = writer

    def run(self):
        scalar_metrics = {
            'eps': FPS(),
            'ep/length': Mean(),
            'ep/reward': Mean(),
        }

        episode = 0
        ep_length = torch.zeros(self.config.workers, device=DEVICE)
        ep_reward = torch.zeros(self.config.workers, device=DEVICE)

        self.model.train()
        s = self.env.reset()

        step = 0
        bar = tqdm(total=self.config.steps, desc='training')
        while step < self.config.steps:
            with torch.no_grad():
                a, _ = self.model(s)
                a = a.sample()

            s_prime, r, d, _ = self.env.step(a)
            self.episode_step(s, a, r, d, s_prime, step=step)
            s = s_prime

            ep_length += 1
            ep_reward += r

            indices, = torch.where(d)
            for i in indices:
                episode += 1

                scalar_metrics['eps'].update(1)
                scalar_metrics['ep/length'].update(ep_length[i].data.cpu().numpy())
                scalar_metrics['ep/reward'].update(ep_reward[i].data.cpu().numpy())

                ep_length[i] = 0
                ep_reward[i] = 0

            for _ in range(self.config.workers):
                step += 1
                bar.update(1)
                if step % self.config.log_interval == 0 and step > 0:
                    for k in scalar_metrics:
                        self.writer.add_scalar(k, scalar_metrics[k].compute_and_reset(), global_step=step)

        bar.close()

    def episode_step(self, *args, **kwargs):
        self.run_callback('on_episode_step', *args, **kwargs)

    def episode_done(self, *args, **kwargs):
        self.run_callback('on_episode_done', *args, **kwargs)


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
    assert config.steps % config.log_interval == 0

    seed_torch(config.seed)
    env = VecEnv([
        lambda: build_transform(gym.make(config.env), config.transforms)
        for _ in range(config.workers)])
    if config.render:
        env = wrappers.TensorboardBatchMonitor(env, config.experiment_path, 10)
    env = wrappers.Torch(env, device=DEVICE)
    env.seed(config.seed)

    writer = SummaryWriter(config.experiment_path)

    model = Model(config.model, env.observation_space, env.action_space)
    model = model.to(DEVICE)
    if config.restore_path is not None:
        model.load_state_dict(torch.load(config.restore_path))

    opt_cb = ActorCriticOptimizationCallback(
        model=model,
        writer=writer,
        config=config)

    if config.rollout.type == 'full_episode':
        rollout_cb = FullEpisodeRolloutCallback(
            config=config,
            callbacks=[opt_cb])
    elif config.rollout.type == 'fixed_horizon':
        rollout_cb = FiniteHorizonRolloutCallback(
            config=config,
            callbacks=[opt_cb])
    else:
        raise AssertionError('invalid config.rollout.type {}'.format(config.rollout.type))

    runner = Runner(
        env=env,
        model=model,
        config=config,
        writer=writer,
        callbacks=[rollout_cb])

    runner.run()

    env.close()


if __name__ == '__main__':
    main()
