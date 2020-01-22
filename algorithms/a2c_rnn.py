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
from algorithms.common import build_optimizer
from model import ModelRNN
from utils import n_step_discounted_return
from vec_env import VecEnv

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: finite horizon undiscounted
# TODO: torch wrapper
# TODO: revisit stat calculation
# TODO: shared weights
# TODO: normalize advantage?


def build_batch(history, state_prime):
    states, actions, rewards, dones = zip(*history)

    states = torch.tensor(states, dtype=torch.float, device=DEVICE).transpose(0, 1)
    actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).transpose(0, 1)
    rewards = torch.tensor(rewards, dtype=torch.float, device=DEVICE).transpose(0, 1)
    dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE).transpose(0, 1)
    state_prime = torch.tensor(state_prime, dtype=torch.float, device=DEVICE)

    return states, actions, rewards, dones, state_prime


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, choices=['momentum', 'rmsprop', 'adam'], default='adam')
    parser.add_argument('--experiment-path', type=str, default='./tf_log/a2c-rnn')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--entropy-weight', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    seed_torch(args.seed)
    env = VecEnv([lambda: gym.make(args.env) for _ in range(args.workers)])
    env.seed(args.seed)
    writer = SummaryWriter(args.experiment_path)

    if args.monitor:
        env = gym.wrappers.Monitor(env, os.path.join('./data', args.env), force=True)

    model = ModelRNN(np.squeeze(env.observation_space.shape), env.action_space.n)
    model = model.to(DEVICE)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.episodes)

    metrics = {
        'loss': Mean(),
        'lr': Last(),
        'ep_length': Mean(),
        'ep_reward': Mean(),
    }

    # ==================================================================================================================
    # training loop
    model.train()
    episode = 0
    ep_length = np.zeros([args.workers])
    ep_reward = np.zeros([args.workers])
    s = env.reset()
    h_p = None
    h_vf = None

    bar = tqdm(total=args.episodes, desc='training')
    while episode < args.episodes:
        history = []

        for _ in range(args.horizon):
            if h_p is not None:
                h_p = [x.detach() for x in h_p]

            a, h = model.policy(torch.tensor(s, dtype=torch.float, device=DEVICE), h_p)
            a = a.sample().data.cpu().numpy()
            s_prime, r, d, _ = env.step(a)
            ep_length += 1
            ep_reward += r
            history.append((s, a, r, d))
            s = s_prime

            indices, = np.where(d)
            for i in indices:
                for tmp in h_p: tmp[:, i] = 0
                for tmp in h_vf: tmp[:, i] = 0
                metrics['ep_length'].update(ep_length[i])
                metrics['ep_reward'].update(ep_reward[i])
                ep_length[i] = 0
                ep_reward[i] = 0
                episode += 1
                bar.update(1)
                scheduler.step()

                if episode % 100 == 0:
                    for k in metrics:
                        writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
                    writer.add_histogram('return', returns, global_step=episode)
                    writer.add_histogram('value', values, global_step=episode)
                    writer.add_histogram('advantage', advantages, global_step=episode)

        states, actions, rewards, dones, state_prime = build_batch(history, s_prime)  # TODO: s or s_prime?

        # critic
        if h_vf is not None:
            h_vf = [x.detach() for x in h_vf]
        values, h_vf = model.value_function(states, h_vf)
        value_prime, _ = model.value_function(state_prime, h_vf)
        returns = n_step_discounted_return(rewards, value_prime.detach(), dones, gamma=args.gamma)
        errors = returns - values
        critic_loss = errors**2

        # actor
        dist, h_p = model.policy(states, h_p)
        advantages = errors.detach()
        actor_loss = -(dist.log_prob(actions) * advantages)
        actor_loss -= args.entropy_weight * dist.entropy()

        loss = (actor_loss + critic_loss).sum(1)

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))

    bar.close()
    env.close()


if __name__ == '__main__':
    main()
