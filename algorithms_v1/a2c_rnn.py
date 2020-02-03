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
from algorithms_v1.common import build_optimizer, transform_env
from algorithms_v1.config import build_default_config
from history import History
from model import ModelRNN
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
            lambda: transform_env(gym.make(config.env), config.transforms)
            for _ in range(config.workers)]),
        device=DEVICE)
    env.seed(config.seed)
    writer = SummaryWriter(config.experiment_path)

    model = ModelRNN(config.model, env.observation_space, env.action_space)
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
    episode = 0
    ep_length = torch.zeros(config.workers, device=DEVICE)
    ep_reward = torch.zeros(config.workers, device=DEVICE)
    s = env.reset()
    h = None

    bar = tqdm(total=config.episodes, desc='training')
    frames = []
    while episode < config.episodes:
        history = History()

        with torch.no_grad():
            for _ in range(config.horizon):
                if frames is not None:
                    frame = torch.tensor(env.render(mode='rgb_array')).permute(2, 0, 1)
                    frames.append(frame)

                a, _, h_prime = model(s.float(), h)
                a = a.sample()
                s_prime, r, d, _ = env.step(a)
                ep_length += 1
                ep_reward += r
                history.append(state=s.float(), action=a, reward=r, done=d)
                s = s_prime
                h = torch.where(d, torch.zeros_like(h_prime), h_prime)

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

        rollout = history.build_rollout(s_prime.float())  # TODO: s or s_prime?
        dist, values = model(rollout.states)
        _, value_prime = model(rollout.state_prime)
        value_prime = value_prime.detach()
        returns = n_step_discounted_return(rollout.rewards, value_prime, rollout.dones, gamma=config.gamma)

        # critic
        errors = returns - values
        critic_loss = errors**2

        # actor
        advantages = errors.detach()
        actor_loss = -(dist.log_prob(rollout.actions) * advantages)
        actor_loss -= config.entropy_weight * dist.entropy()

        loss = (actor_loss + critic_loss).sum(1)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))
        metrics['step/entropy'].update(dist.entropy().data.cpu().numpy())

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()

    bar.close()
    env.close()


if __name__ == '__main__':
    main()