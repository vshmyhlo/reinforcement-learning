import click
import gym
import gym_minigrid
import torch
from all_the_tools.config import Config as C
from all_the_tools.meters import Stack
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils
import wrappers
from agent import Agent
from history import History
from vec_env_serial import VecEnv

torch.autograd.set_detect_anomaly(True)

# TODO: log dist of resets on each step
# TODO: seed
# TODO: tests
# TODO: normalize obs
# TODO: Don’t forget to normalize observations. Everywhere that observations might be being used. 3
# TODO: check BPTT, check detach of memory
# TODO: TBPTT(k1, k2): k1 - steps forward, k2 - steps backward
"""

 a backward pass through the most recent h time steps is performed anew each time the network is run through an additional time step
 one may consider letting the network run through h0 additional time steps before performing the next BPTT computation, where h0 <= h.
"""

# TODO: average reward
# TODO: obs normalization
# TODO: critic loss scale

# TODO: log training steps
# TODO: log finished eps as a function of opt steps
# TODO: how average reward is derived


@click.command()
@click.option("--experiment-path", type=click.Path(), required=True)
def main(**kwargs):
    config = C(
        random_seed=42,
        learning_rate=1e-4,
        horizon=32,
        discount=0.99,
        num_episodes=100000,
        num_workers=32,
        entropy_weight=1e-2,
        episode_log_interval=100,
        opt_log_interval=10,
        average_reward_lr=0.001,
    )
    for k in kwargs:
        config[k] = kwargs[k]

    utils.random_seed(config.random_seed)
    writer = SummaryWriter(config.experiment_path)

    # build env
    env = VecEnv([build_env for _ in range(config.num_workers)])
    env = wrappers.TensorboardBatchMonitor(
        env, writer, log_interval=config.episode_log_interval, fps_mul=0.5
    )
    env = wrappers.Torch(env)

    # build agent and optimizer
    agent = Agent(env.observation_space, env.action_space)
    optimizer = torch.optim.Adam(
        agent.parameters(), config.learning_rate * config.num_workers, betas=(0.0, 0.999)
    )
    average_reward = 0

    # train
    metrics = {
        "episode/return": Stack(),
        "episode/length": Stack(),
        "rollout/reward": Stack(),
        "rollout/value_target": Stack(),
        "rollout/value": Stack(),
        "rollout/td_error": Stack(),
        "rollout/entropy": Stack(),
        "rollout/actor_loss": Stack(),
        "rollout/critic_loss": Stack(),
        "rollout/loss": Stack(),
    }

    episode = 0
    opt_step = 0
    pbar = tqdm(total=config.num_episodes)

    env.seed(config.random_seed)
    obs = env.reset()
    action = torch.zeros(config.num_workers, dtype=torch.int)
    memory = agent.zero_memory(config.num_workers)

    while episode < config.num_episodes:
        history = History()
        memory = tuple(x.detach() for x in memory)

        for i in range(config.horizon):
            transition = history.append_transition()

            dist, value, memory_prime = agent(obs, action, memory)
            transition.record(value=value, entropy=dist.entropy())
            action = select_action(dist)
            transition.record(log_prob=dist.log_prob(action))

            obs_prime, reward, done, info = env.step(action)
            transition.record(reward=reward, done=done)
            memory_prime = agent.reset_memory(memory_prime, done)

            obs, memory = obs_prime, memory_prime

            for i in info:
                if "episode" not in i:
                    continue
                episode += 1

                metrics["episode/return"].update(i["episode"]["r"])
                metrics["episode/length"].update(i["episode"]["l"])

                if episode % config.episode_log_interval == 0:
                    print("log episode")

                    for k in [
                        "episode/return",
                        "episode/length",
                    ]:
                        v = metrics[k].compute_and_reset()
                        writer.add_scalar(f"{k}/mean", v.mean(), global_step=episode)
                        writer.add_histogram(f"{k}/hist", v, global_step=episode)

                    writer.flush()

                pbar.update()

        rollout = history.build()

        _, value_prime, _ = agent(obs_prime, action, memory_prime)

        value_target = utils.n_step_bootstrapped_return(
            reward_t=rollout.reward,
            done_t=rollout.done,
            value_prime=value_prime.detach(),
            discount=config.discount,
        )

        # advantage = utils.generalized_advantage_estimation(
        #     reward_t=rollout.reward,
        #     value_t=rollout.value.detach(),
        #     value_prime=value_prime.detach(),
        #     done_t=rollout.done,
        #     gamma=config.discount,
        #     lambda_=0.96,
        # )
        # value_target = advantage + rollout.value.detach()

        # value_target = utils.differential_n_step_bootstrapped_return(
        #     reward_t=rollout.reward,
        #     done_t=rollout.done,
        #     value_prime=value_prime.detach(),
        #     average_reward=average_reward,
        # )

        td_error = value_target - rollout.value

        critic_loss = 0.5 * td_error.pow(2)
        actor_loss = (
            -rollout.log_prob * td_error.detach() - config.entropy_weight * rollout.entropy
        )
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.sum(1).mean().backward()
        optimizer.step()
        average_reward += config.average_reward_lr * td_error.detach().sum(1).mean()
        opt_step += 1

        metrics["rollout/reward"].update(rollout.reward.detach())
        metrics["rollout/value"].update(rollout.value.detach())
        metrics["rollout/value_target"].update(value_target.detach())
        metrics["rollout/td_error"].update(td_error.detach())
        metrics["rollout/entropy"].update(rollout.entropy.detach())
        metrics["rollout/actor_loss"].update(actor_loss.detach())
        metrics["rollout/critic_loss"].update(critic_loss.detach())
        metrics["rollout/loss"].update(loss.detach())

        if opt_step % config.opt_log_interval == 0:
            # td_error_std_normalized = td_error.std() / value_target.std()
            print("log rollout")

            writer.add_scalar("rollout/average_reward", average_reward, global_step=episode)
            grad_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0) for p in agent.parameters()]), 2.0
            )
            writer.add_scalar("rollout/grad_norm", grad_norm, global_step=episode)

            for k in [
                "rollout/reward",
                "rollout/value_target",
                "rollout/value",
                "rollout/td_error",
            ]:
                v = metrics[k].compute_and_reset()
                writer.add_scalar(f"{k}/mean", v.mean(), global_step=episode)
                writer.add_histogram(f"{k}/hist", v, global_step=episode)

            for k in [
                "rollout/entropy",
                "rollout/actor_loss",
                "rollout/critic_loss",
                "rollout/loss",
            ]:
                v = metrics[k].compute_and_reset()
                writer.add_scalar(f"{k}/mean", v.mean(), global_step=episode)

            writer.flush()

            # writer.add_scalar(
            #     "rollout/td_error_std_normalized", td_error_std_normalized, global_step=episode
            # )
            # writer.add_histogram("rollout/reward", rollout.reward, global_step=episode)

    env.close()
    writer.close()


def build_env():
    def scale_reward(r):
        return r
        # return r * 5
        # return r * 10 - 0.02

    # env = gym.make("CartPole-v1")
    env = "MiniGrid-Empty-Random-6x6-v0"
    # env = "MiniGrid-FourRooms-v0"
    # env = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    env = gym.make(env)
    env = wrappers.RandomFirstReset(env, 256)
    env = gym_minigrid.wrappers.OneHotPartialObsWrapper(env)
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    env = gym.wrappers.TransformReward(env, scale_reward)
    env.reward_range = tuple(map(scale_reward, env.reward_range))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def select_action(dist: torch.distributions.Distribution):
    return dist.sample()


if __name__ == "__main__":
    main()
