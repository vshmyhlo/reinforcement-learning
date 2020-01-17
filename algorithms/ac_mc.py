import itertools
import os

import gym
import numpy as np
import torch
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from ticpfptp.metrics import Mean
from tqdm import tqdm

import utils
from model import Model
from model import PolicyCategorical, ValueFunction
from utils import total_return


# TODO: train/eval
# TODO: bn update
# TODO: return normalization
# TODO: monitored session
# TODO: normalize advantage?


def build_batch(history):
    states, actions, rewards = zip(*history)

    states = torch.tensor(states).transpose(0, 1).float()
    actions = torch.tensor(actions).transpose(0, 1)
    rewards = torch.tensor(rewards).transpose(0, 1).float()

    return states, actions, rewards


def build_optimizer(optimizer, parameters, learning_rate):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, learning_rate, weight_decay=1e-4)
    elif optimizer == 'momentum':
        return torch.optim.SGD(parameters, learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        raise AssertionError('invalid optimizer {}'.format(optimizer))


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'momentum'], default='adam')
    parser.add_argument('--experiment-path', type=str, default='./tf_log/ac-mc')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--entropy-weight', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    seed_torch(args.seed)
    env = gym.make(args.env)
    env.seed(args.seed)
    writer = SummaryWriter(args.experiment_path)

    if args.monitor:
        env = gym.wrappers.Monitor(env, os.path.join('./data', args.env), force=True)

    model = Model(
        policy=PolicyCategorical(np.squeeze(env.observation_space.shape), env.action_space.n),
        value_function=ValueFunction(np.squeeze(env.observation_space.shape)))
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.learning_rate)

    metrics = {
        'loss': Mean(),
        'ep_length': Mean(),
        'ep_reward': Mean(),
    }

    # ==================================================================================================================
    # training loop
    model.train()
    for episode in tqdm(range(args.episodes), desc='training'):
        history = []
        s = env.reset()
        ep_reward = 0

        with torch.no_grad():
            for ep_length in itertools.count():
                a = model.policy(torch.tensor(s).float()).sample().item()
                s_prime, r, d, _ = env.step(a)
                ep_reward += r
                history.append(([s], [a], [r]))

                if d:
                    break
                else:
                    s = s_prime

        states, actions, rewards = build_batch(history)

        # critic
        values = model.value_function(states)
        returns = total_return(rewards, gamma=args.gamma)
        errors = returns - values
        critic_loss = errors**2

        # actor
        dist = model.policy(states)
        advantages = errors.detach()
        actor_loss = -(dist.log_prob(actions) * advantages)
        actor_loss -= args.entropy_weight * dist.entropy()

        loss = (actor_loss + critic_loss).mean(1)

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['ep_length'].update(ep_length)
        metrics['ep_reward'].update(ep_reward)

        if episode % 100 == 0:
            for k in metrics:
                writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)


if __name__ == '__main__':
    main()
