import itertools
import os

import gym
import gym.wrappers
import numpy as np
import torch
from all_the_tools.metrics import Mean, Last
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils
import wrappers
from algorithms.common import build_optimizer
from model import Model, ModelShared
from utils import total_discounted_return

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: make shared weights work
# TODO: normalize advantage
# TODO: train/eval
# TODO: bn update
# TODO: return normalization
# TODO: monitored session
# TODO: normalize advantage?


def build_batch(history):
    states, actions, rewards = zip(*history)

    states = torch.stack(states, 1)
    actions = torch.stack(actions, 1)
    rewards = torch.stack(rewards, 1)

    return states, actions, rewards


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, choices=['momentum', 'rmsprop', 'adam'], default='adam')
    parser.add_argument('--experiment-path', type=str, default='./tf_log/pg-mc')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--entropy-weight', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    seed_torch(args.seed)
    env = wrappers.Torch(gym.make(args.env), device=DEVICE)
    env.seed(args.seed)
    writer = SummaryWriter(args.experiment_path)

    if args.monitor:
        env = gym.wrappers.Monitor(env, os.path.join('./data', args.env), force=True)

    if args.shared:
        model = ModelShared(env.observation_space.shape, env.action_space.n)
    else:
        model = Model(env.observation_space.shape, env.action_space.n)
    model = model.to(DEVICE)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.episodes)

    metrics = {
        'loss': Mean(),
        'lr': Last(),
        'ep/length': Mean(),
        'ep/reward': Mean(),
        'step/entropy': Mean(),
    }

    # ==================================================================================================================
    # training loop
    model.train()
    for episode in tqdm(range(args.episodes), desc='training'):
        history = []
        frames = [] if episode % args.log_interval == 0 else None
        s = env.reset()
        ep_reward = 0

        with torch.no_grad():
            for ep_length in itertools.count():
                if frames is not None:
                    frame = torch.tensor(env.render(mode='rgb_array')).permute(2, 0, 1)
                    frames.append(frame)

                a, _ = model(s.float())
                a = a.sample()
                s_prime, r, d, _ = env.step(a)
                ep_reward += r
                history.append((s.float().unsqueeze(0), a.unsqueeze(0), r.unsqueeze(0)))

                if d:
                    break
                else:
                    s = s_prime

            states, actions, rewards = build_batch(history)

        dist, _ = model(states)

        # actor
        returns = total_discounted_return(rewards, gamma=args.gamma)
        advantages = returns.detach()
        actor_loss = -(dist.log_prob(actions) * advantages)
        actor_loss -= args.entropy_weight * dist.entropy()

        loss = actor_loss.sum(1)

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))
        metrics['ep/length'].update(ep_length)
        metrics['ep/reward'].update(ep_reward.data.cpu().numpy())
        metrics['step/entropy'].update(dist.entropy().data.cpu().numpy())

        if episode % args.log_interval == 0:
            for k in metrics:
                writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
            writer.add_histogram('step/action', actions, global_step=episode)
            writer.add_histogram('step/reward', rewards, global_step=episode)
            writer.add_histogram('step/return', returns, global_step=episode)
            writer.add_histogram('step/advantage', advantages, global_step=episode)
            writer.add_video('episode', torch.stack(frames, 0).unsqueeze(0), fps=24, global_step=episode)


if __name__ == '__main__':
    main()
