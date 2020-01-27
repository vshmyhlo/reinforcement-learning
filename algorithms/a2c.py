import os

import gym
import numpy as np
import torch
import torch.optim
import torch.optim
from all_the_tools.metrics import Mean, Last
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils
import wrappers
from algorithms.common import build_optimizer
from model import Model, ModelShared
from utils import n_step_discounted_return
from vec_env import VecEnv

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: finite horizon undiscounted
# TODO: torch wrapper
# TODO: revisit stat calculation
# TODO: shared weights
# TODO: normalize advantage?
# TODO: normalize input (especially images)


def build_batch(history, state_prime):
    states, actions, rewards, dones = zip(*history)

    states = torch.stack(states, 1)
    actions = torch.stack(actions, 1)
    rewards = torch.stack(rewards, 1)
    dones = torch.stack(dones, 1)

    return states, actions, rewards, dones, state_prime


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, choices=['momentum', 'rmsprop', 'adam'], default='adam')
    parser.add_argument('--experiment-path', type=str, default='./tf_log/a2c')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--entropy-weight', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    seed_torch(args.seed)
    env = wrappers.Torch(
        VecEnv([lambda: gym.make(args.env) for _ in range(args.workers)]),
        device=DEVICE)
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
    episode = 0
    ep_length = torch.zeros(args.workers, device=DEVICE)
    ep_reward = torch.zeros(args.workers, device=DEVICE)
    s = env.reset()

    bar = tqdm(total=args.episodes, desc='training')
    while episode < args.episodes:
        history = []

        with torch.no_grad():
            for _ in range(args.horizon):
                a, _ = model(s.float())
                a = a.sample()
                s_prime, r, d, _ = env.step(a)
                ep_length += 1
                ep_reward += r
                history.append((s.float(), a, r, d))
                s = s_prime

                indices, = torch.where(d)
                for i in indices:
                    metrics['ep/length'].update(ep_length[i].data.cpu().numpy())
                    metrics['ep/reward'].update(ep_reward[i].data.cpu().numpy())
                    ep_length[i] = 0
                    ep_reward[i] = 0
                    episode += 1
                    scheduler.step()
                    bar.update(1)

                    if episode % args.log_interval == 0:
                        for k in metrics:
                            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
                        writer.add_histogram('step/action', actions, global_step=episode)
                        writer.add_histogram('step/reward', rewards, global_step=episode)
                        writer.add_histogram('step/return', returns, global_step=episode)
                        writer.add_histogram('step/value', values, global_step=episode)
                        writer.add_histogram('step/advantage', advantages, global_step=episode)

            states, actions, rewards, dones, state_prime = build_batch(history, s_prime.float())  # TODO: s or s_prime?

        dist, values = model(states)
        _, value_prime = model(state_prime)
        value_prime = value_prime.detach()

        # critic
        returns = n_step_discounted_return(rewards, value_prime, dones, gamma=args.gamma)
        errors = returns - values
        critic_loss = errors**2

        # actor
        advantages = errors.detach()
        actor_loss = -(dist.log_prob(actions) * advantages)
        actor_loss -= args.entropy_weight * dist.entropy()

        loss = (actor_loss + critic_loss).sum(1)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))
        metrics['step/entropy'].update(dist.entropy().data.cpu().numpy())

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    bar.close()
    env.close()


if __name__ == '__main__':
    main()
