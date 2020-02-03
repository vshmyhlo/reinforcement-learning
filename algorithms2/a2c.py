from history import History


class Observable(object):

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def run_callback(self, name, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, name)(*args, **kwargs)


class RolloutCallback(Observable):
    def __init__(self, callbacks):
        super(Observable, self).__init__(callbacks)

        self.history = History()

    def on_episode_step(self, s, a, r, d, s_prime):
        self.history.append(state=s, action=a, reward=r, done=d)

        if len(self.history) == self.config.horizon:
            rollout = self.history.build_rollout(s_prime)  # TODO: s or s_prime?
            self.rollout_done(rollout)
            self.history = History()

    def rollout_done(self, *args, **kwargs):
        self.run_callback('on_rollout_done', *args, **kwargs)


class OptimizationCallback(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = build_optimizer(self.config.opt, model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.episodes)

    def on_rollout_done(self, rollout):
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

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(self.scheduler.get_lr()))
        metrics['step/entropy'].update(dist.entropy().data.cpu().numpy())

        # training
        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()


class Runner(Observable):
    def __init__(self, env, model, callbacks):
        super(Observable, self).__init__(callbacks)

        self.env = env
        self.model = model

    def run(self):
        self.model.train()
        episode = 0
        ep_length = torch.zeros(self.config.workers, device=DEVICE)
        ep_reward = torch.zeros(self.config.workers, device=DEVICE)
        s = self.env.reset()

        bar = tqdm(total=self.config.episodes, desc='training')
        frames = []
        while episode < self.config.episodes:
            if frames is not None:
                frame = torch.tensor(self.env.render(mode='rgb_array')).permute(2, 0, 1)
                frames.append(frame)

            with torch.no_grad():
                a, _ = self.model(s)
                a = a.sample()

            s_prime, r, d, _ = self.env.step(a)
            ep_length += 1
            ep_reward += r
            self.episode_step(s, a, r, d, s_prime)
            s = s_prime

            indices, = torch.where(d)
            for i in indices:
                metrics['eps'].update(1)
                metrics['ep/length'].update(ep_length[i].data.cpu().numpy())
                metrics['ep/reward'].update(ep_reward[i].data.cpu().numpy())
                ep_length[i] = 0
                ep_reward[i] = 0
                episode += 1
                scheduler.step()
                bar.update(1)

                if episode % config.log_interval == 0 and episode > 0:
                    for k in metrics:
                        writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
                    writer.add_histogram('step/action', rollout.actions, global_step=episode)
                    writer.add_histogram('step/reward', rollout.rewards, global_step=episode)
                    writer.add_histogram('step/return', returns, global_step=episode)
                    writer.add_histogram('step/value', values, global_step=episode)
                    writer.add_histogram('step/advantage', advantages, global_step=episode)

                if episode % 1000 == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(config.experiment_path, 'model_{}.pth'.format(episode)))

                if i == 0:
                    if frames is not None:
                        writer.add_video(
                            'episode', torch.stack(frames, 0).unsqueeze(0), fps=24, global_step=episode)
                    frames = []

    def episode_step(self, *args, **kwargs):
        self.run_callback('on_episode_step', *args, **kwargs)

    def episode_done(self, *args, **kwargs):
        self.run_callback('on_episode_done', *args, **kwargs)


def main():
    env = ...
    model = ...

    opt_cb = OptimizationCallback(model=model)
    rollout_cb = RolloutCallback(callbacks=[opt_cb])
    runner = Runner(env=env, model=model, callbacks=[rollout_cb])

    runner.run()


########################################################################################################################

import argparse

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
from algorithms.common import build_optimizer, build_transform
from config import build_default_config
from model import Model
from utils import n_step_discounted_return
from vec_env import VecEnv

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: torch wrapper
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


def main():
    args = build_parser().parse_args()
    config = build_default_config()
    config.merge_from_file(args.config_path)
    config.experiment_path = args.experiment_path
    config.restore_path = args.restore_path
    config.freeze()
    del args

    seed_torch(config.seed)
    env = wrappers.Torch(
        VecEnv([
            lambda: build_transform(gym.make(config.env), config.transforms)
            for _ in range(config.workers)]),
        device=DEVICE)
    env.seed(config.seed)
    writer = SummaryWriter(config.experiment_path)

    model = Model(config.model, env.observation_space, env.action_space)
    model = model.to(DEVICE)
    if config.restore_path is not None:
        model.load_state_dict(torch.load(config.restore_path))

    metrics = {
        'loss': Mean(),
        'lr': Last(),
        'eps': FPS(),
        'ep/length': Mean(),
        'ep/reward': Mean(),
        'step/entropy': Mean(),
    }

    # loop

    bar.close()
    env.close()


if __name__ == '__main__':
    main()
