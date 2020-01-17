import os

import gym
import numpy as np
import torch
from all_the_tools.metrics import Mean
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils
from algorithms.model import Model
from algorithms.model import PolicyCategorical, ValueFunction
from algorithms.utils import n_step_return
from vec_env import VecEnv


# TODO: revisit stat calculation
# TODO: normalize advantage?


def build_batch(history, state_prime):
    states, actions, rewards, dones = zip(*history)

    states = torch.tensor(states).transpose(0, 1).float()
    actions = torch.tensor(actions).transpose(0, 1)
    rewards = torch.tensor(rewards).transpose(0, 1).float()
    dones = torch.tensor(np.uint8(dones)).transpose(0, 1)
    state_prime = torch.tensor(state_prime).float()

    return states, actions, rewards, dones, state_prime


def build_optimizer(optimizer, parameters, learning_rate):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, learning_rate, weight_decay=1e-4)
    elif optimizer == 'momentum':
        return torch.optim.SGD(parameters, learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        raise AssertionError('invalid optimizer {}'.format(optimizer))


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'momentum'], default='adam')
    parser.add_argument('--experiment-path', type=str, default='./tf_log/a2c')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--entropy-weight', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
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
    episode = 0
    ep_length = np.zeros([args.workers])
    ep_reward = np.zeros([args.workers])
    s = env.reset()

    bar = tqdm(total=args.episodes, desc='training')
    while episode < args.episodes:
        history = []

        for _ in range(args.horizon):
            a = model.policy(torch.tensor(s).float()).sample().data.cpu().numpy()
            s_prime, r, d, _ = env.step(a)
            ep_length += 1
            ep_reward += r
            history.append((s, a, r, d))
            s = s_prime

            for i in range(args.workers):
                if d[i]:
                    metrics['ep_length'].update(ep_length[i])
                    metrics['ep_reward'].update(ep_reward[i])
                    ep_length[i] = 0
                    ep_reward[i] = 0
                    episode += 1
                    bar.update(1)

                    if episode % 100 == 0:
                        for k in metrics:
                            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)

        states, actions, rewards, dones, state_prime = build_batch(history, s_prime)  # TODO: s or s_prime?

        # critic
        values = model.value_function(states)
        value_prime = model.value_function(state_prime).detach()
        returns = n_step_return(rewards, value_prime, dones, gamma=args.gamma)
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

    bar.close()
    env.close()


if __name__ == '__main__':
    main()
