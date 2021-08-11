import os

import click
import gym
import gym.wrappers
import numpy as np
import pybulletgym
import torch
import torch.nn as nn
import torch.optim
from all_the_tools.config import load_config
from all_the_tools.metrics import FPS, Last, Mean
from all_the_tools.torch.utils import seed_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wrappers
from algo.common import build_env, build_optimizer
from history import History
from model import Model
from utils import n_step_bootstrapped_return
from vec_env_parallel import VecEnv

pybulletgym

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO: check how finished episodes count
# TODO: revisit stat calculation
# TODO: normalize advantage?
# TODO: normalize input (especially images)
# TODO: refactor EPS (noisy and incorrect statistics)
# TODO: sum or average entropy of each action


@click.command()
@click.option("--config-path", type=click.Path(), required=True)
@click.option("--experiment-path", type=click.Path(), required=True)
@click.option("--restore-path", type=click.Path())
@click.option("--render", is_flag=True)
def main(config_path, **kwargs):
    config = load_config(config_path, **kwargs)
    del config_path, kwargs

    writer = SummaryWriter(config.experiment_path)

    seed_torch(config.seed)
    env = VecEnv([lambda: build_env(config) for _ in range(config.workers)])
    if config.render:
        env = wrappers.TensorboardBatchMonitor(env, writer, config.log_interval)
    env = wrappers.Torch(env, dtype=torch.float, device=DEVICE)
    env.seed(config.seed)

    model = Model(config.model, env.observation_space, env.action_space)
    model = model.to(DEVICE)
    if config.restore_path is not None:
        model.load_state_dict(torch.load(config.restore_path))
    optimizer = build_optimizer(config.opt, model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.episodes)

    metrics = {
        "loss": Mean(),
        "lr": Last(),
        "eps": FPS(),
        "ep/length": Mean(),
        "ep/return": Mean(),
        "rollout/entropy": Mean(),
    }

    # ==================================================================================================================
    # training loop
    model.train()
    episode = 0
    s = env.reset()

    bar = tqdm(total=config.episodes, desc="training")
    while episode < config.episodes:
        history = History()

        with torch.no_grad():
            for _ in range(config.horizon):
                a, _ = model(s)
                a = a.sample()
                s_prime, r, d, info = env.step(a)
                history.append(state=s, action=a, reward=r, done=d, state_prime=s_prime)
                s = s_prime

                (indices,) = torch.where(d)
                for i in indices:
                    metrics["eps"].update(1)
                    metrics["ep/length"].update(info[i]["episode"]["l"])
                    metrics["ep/return"].update(info[i]["episode"]["r"])
                    episode += 1
                    scheduler.step()
                    bar.update(1)

                    if episode % config.log_interval == 0 and episode > 0:
                        for k in metrics:
                            writer.add_scalar(
                                k, metrics[k].compute_and_reset(), global_step=episode
                            )
                        writer.add_histogram(
                            "rollout/action", rollout.actions, global_step=episode
                        )
                        writer.add_histogram(
                            "rollout/reward", rollout.rewards, global_step=episode
                        )
                        writer.add_histogram("rollout/return", returns, global_step=episode)
                        writer.add_histogram("rollout/value", values, global_step=episode)
                        writer.add_histogram("rollout/advantage", advantages, global_step=episode)

                        torch.save(
                            model.state_dict(),
                            os.path.join(config.experiment_path, "model_{}.pth".format(episode)),
                        )

        rollout = history.full_rollout()
        dist, values = model(rollout.states)
        with torch.no_grad():
            _, value_prime = model(rollout.states_prime[:, -1])
            returns = n_step_bootstrapped_return(
                rollout.rewards, value_prime, rollout.dones, gamma=config.gamma
            )

        # critic
        errors = returns - values
        critic_loss = errors ** 2

        # actor
        advantages = errors.detach()
        log_prob = dist.log_prob(rollout.actions)
        entropy = dist.entropy()

        if isinstance(env.action_space, gym.spaces.Box):
            log_prob = log_prob.sum(-1)
            entropy = entropy.sum(-1)
        assert log_prob.dim() == entropy.dim() == 2

        actor_loss = -log_prob * advantages - config.entropy_weight * entropy

        # loss
        loss = (actor_loss + 0.5 * critic_loss).mean(1)

        metrics["loss"].update(loss.data.cpu().numpy())
        metrics["lr"].update(np.squeeze(scheduler.get_lr()))
        metrics["rollout/entropy"].update(dist.entropy().data.cpu().numpy())

        # training
        optimizer.zero_grad()
        loss.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    bar.close()
    env.close()


if __name__ == "__main__":
    main()
