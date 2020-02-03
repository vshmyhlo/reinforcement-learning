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
            lambda: gym.wrappers.TransformObservation(
                wrappers.StackObservation(gym.make(config.env), 4),
                transform_env(config.transform))
            for _ in range(config.workers)]),
        device=DEVICE)
    env.seed(config.seed)
    writer = SummaryWriter(config.experiment_path)

    model = Model(config.model, env.observation_space, env.action_space)
    model = model.to(DEVICE)
    if config.restore_path is not None:
        model.load_state_dict(torch.load(config.restore_path))
    optimizer = build_optimizer(config.opt, model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.steps)

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
    ep_length = torch.zeros(config.workers, device=DEVICE)
    ep_reward = torch.zeros(config.workers, device=DEVICE)
    s = env.reset()

    history = History()
    frames = []
    for step in tqdm(range(config.steps), desc='training'):
        with torch.no_grad():
            if frames is not None:
                frame = torch.tensor(env.render(mode='rgb_array')).permute(2, 0, 1)
                frames.append(frame)

            a, _ = model(s.float())
            a = a.sample()
            s_prime, r, d, _ = env.step(a)
            ep_length += 1
            ep_reward += r
            history.append(state=s.float(), action=a, reward=r, done=d)
            s = s_prime

            indices, = torch.where(d)
            for i in indices:
                metrics['eps'].update(1)
                metrics['ep/length'].update(ep_length[i].data.cpu().numpy())
                metrics['ep/reward'].update(ep_reward[i].data.cpu().numpy())
                ep_length[i] = 0
                ep_reward[i] = 0

                if i == 0:
                    if frames is not None:
                        writer.add_video(
                            'episode', torch.stack(frames, 0).unsqueeze(0), fps=24, global_step=step)
                    frames = []

        if len(history) == config.horizon:
            rollout = history.build_rollout(s_prime.float())  # TODO: s or s_prime?
            history = History()

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
            scheduler.step()

        if step % config.log_interval == 0 and step > 0:
            for k in metrics:
                writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=step)
            writer.add_histogram('step/action', rollout.actions, global_step=step)
            writer.add_histogram('step/reward', rollout.rewards, global_step=step)
            writer.add_histogram('step/return', returns, global_step=step)
            writer.add_histogram('step/value', values, global_step=step)
            writer.add_histogram('step/advantage', advantages, global_step=step)

            torch.save(
                model.state_dict(),
                os.path.join(config.experiment_path, 'model_{}.pth'.format(step)))

    env.close()


if __name__ == '__main__':
    main()
