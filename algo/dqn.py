import argparse

import gym_minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from all_the_tools.metrics import Mean, FPS, Last
from all_the_tools.torch.utils import seed_torch, one_hot
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wrappers
import wrappers.torch
from algo.common import build_optimizer, build_env
from history import History
from model import ModelDQN
from utils import one_step_discounted_return
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
    parser.add_argument('--no-render', action='store_true')

    return parser


def sample_action(action_value, e):
    greedy = one_hot(action_value.argmax(-1), action_value.size(-1))
    random = torch.full_like(greedy, 1 / action_value.size(-1))

    probs = torch.where(
        torch.empty(greedy.size(0), 1, device=greedy.device).uniform_() > e,
        greedy,
        random)

    dist = torch.distributions.Categorical(probs=probs)

    return dist.sample()


def main():
    args = build_parser().parse_args()
    config = build_default_config()
    config.merge_from_file(args.config_path)
    config.experiment_path = args.experiment_path
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
    env = wrappers.torch.Torch(env, device=DEVICE)
    env.seed(config.seed)

    policy_model = ModelDQN(config.model, env.observation_space, env.action_space).to(DEVICE)
    target_model = ModelDQN(config.model, env.observation_space, env.action_space).to(DEVICE)
    target_model.load_state_dict(policy_model.state_dict())
    optimizer = build_optimizer(config.opt, policy_model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.episodes)

    metrics = {
        'loss': Mean(),
        'lr': Last(),
        'eps': FPS(),
        'ep/length': Mean(),
        'ep/reward': Mean(),
    }

    # ==================================================================================================================
    # training loop
    policy_model.train()
    target_model.eval()
    episode = 0
    s = env.reset()
    e_base = 0.95
    e_step = np.exp(np.log(0.05 / e_base) / config.episodes)

    bar = tqdm(total=config.episodes, desc='training')
    history = History()
    while episode < config.episodes:
        with torch.no_grad():
            for _ in range(config.horizon):
                av = policy_model(s)
                a = sample_action(av, e_base * e_step**episode)
                s_prime, r, d, meta = env.step(a)
                history.append(state=s.cpu(), action=a.cpu(), reward=r.cpu(), done=d.cpu(), state_prime=s_prime.cpu())
                # history.append(state=s, action=a, reward=r, done=d, state_prime=s_prime)
                s = s_prime

                indices, = torch.where(d)
                for i in indices:
                    metrics['eps'].update(1)
                    metrics['ep/length'].update(meta[i]['episode']['l'])
                    metrics['ep/reward'].update(meta[i]['episode']['r'])
                    episode += 1
                    scheduler.step()
                    bar.update(1)

                    if episode % 10 == 0:
                        target_model.load_state_dict(policy_model.state_dict())

                    if episode % config.log_interval == 0 and episode > 0:
                        for k in metrics:
                            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=episode)
                        writer.add_scalar('e', e_base * e_step**episode, global_step=episode)
                        writer.add_histogram('rollout/action', rollout.actions, global_step=episode)
                        writer.add_histogram('rollout/reward', rollout.rewards, global_step=episode)
                        writer.add_histogram('rollout/return', returns, global_step=episode)
                        writer.add_histogram('rollout/action_value', action_values, global_step=episode)

        rollout = history.full_rollout()
        action_values = policy_model(rollout.states)
        action_values = action_values * one_hot(rollout.actions, action_values.size(-1))
        action_values = action_values.sum(-1)
        with torch.no_grad():
            action_values_prime = target_model(rollout.states_prime)
            action_values_prime, _ = action_values_prime.detach().max(-1)
        returns = one_step_discounted_return(rollout.rewards, action_values_prime, rollout.dones, gamma=config.gamma)

        # critic
        errors = returns - action_values
        critic_loss = errors**2

        loss = (critic_loss * 0.5).mean(1)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        nn.utils.clip_grad_norm_(policy_model.parameters(), 0.5)
        optimizer.step()

    bar.close()
    env.close()


if __name__ == '__main__':
    main()
