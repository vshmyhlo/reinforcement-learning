import click
import gym
import gym_minigrid
import torch
import torch.nn as nn
from all_the_tools.config import Config as C
from all_the_tools.meters import Stack
from tensorboardX import SummaryWriter
from tqdm import tqdm

import envs
import utils
import wrappers
from agent import Agent
from history import History
from vec_env_serial import VecEnv

torch.autograd.set_detect_anomaly(True)

# TODO: test memory by remembering sequence of states


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


"""
Gradients are additionally clipped per parameter to be within between ±5√v where v is the running estimate of the second moment of the (unclipped) gradient
PPO
Because we use GAE with λ = 0.95, the GAE rewards need to be smoothed over a number of timesteps  1/λ = 20; using 256 timesteps causes relatively little loss.
increasing the batch size
1.0 value loss weight
lower entropy
adam momentum
normalize rewards
We normalize rewards using a running estimate of the standard deviation, and the value loss weight is applied post-normalization
larger GAE horizon and multiple lstm rollouts per sequence
All float observations (including booleans which are treated as floats that happen to take values 0 or 1) are normalized before feeding into the neural network.
For each observation, we keep a running mean and standard deviation of all data ever observed; at each timestep we subtract the mean and divide by the st dev, clipping the final result to be within (-5, 5).
"""

# TODO: check batch-size aggregation and learning-rate
# TODO: plot over opt steps, not episodes
# TODO: data staleness and sample-reuse
# TODO: how LSTM hidden state is stored in rollouts and how it is used for optimisation
# TODO: test multi-agent communication
# TODO: do evaluation run every n steps

# TODO: log over samples processed to emphasize sample-efficiency
# TODO: use graph-generative model for program synthesis


@click.command()
@click.option("--experiment-path", type=click.Path(), required=True)
def main(**kwargs):
    config = C(
        random_seed=42,
        learning_rate=1e-3,
        horizon=16,
        discount=0.995,
        num_episodes=10000,
        num_workers=256,
        entropy_weight=1e-2,
        episode_log_interval=100,
        opt_log_interval=10,
        average_reward_lr=0.001,
        clip_grad_norm=None,
        model=C(
            num_features=64,
            encoder=C(
                type="minigrid",
            ),
            memory=C(
                type="lstm",
            ),
        ),
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
    agent = Agent(
        env.observation_space,
        env.action_space,
        **config.model,
    )
    optimizer = torch.optim.Adam(
        agent.parameters(),
        config.learning_rate,
        betas=(0.0, 0.999),
    )
    average_reward = 0

    # load state
    # state = torch.load("./state.pth")
    # agent.load_state_dict(state["agent"])
    # optimizer.load_state_dict(state["optimizer"])

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

    # r_stats = utils.RunningStats()

    while episode < config.num_episodes:
        history = History()
        memory = agent.detach_memory(memory)

        for i in range(config.horizon):
            transition = history.append_transition()

            dist, value, memory_prime = agent(obs, action, memory)
            transition.record(value=value, entropy=dist.entropy())
            action = select_action(dist)
            transition.record(log_prob=dist.log_prob(action))

            obs_prime, reward, done, info = env.step(action)
            # for r in reward:
            #     r_stats.push(r)
            # reward = reward / r_stats.standard_deviation()
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

                    # for k in [
                    #     "episode/return",
                    #     "episode/length",
                    # ]:
                    #     v = metrics[k].compute_and_reset()
                    #     writer.add_scalar(f"{k}/mean", v.mean(), global_step=episode)
                    #     writer.add_histogram(f"{k}/hist", v, global_step=episode)

                    writer.flush()

                pbar.update()

        rollout = history.build()

        _, value_prime, _ = agent(obs_prime, action, memory_prime)

        # value_target = utils.n_step_bootstrapped_return(
        #     reward_t=rollout.reward,
        #     done_t=rollout.done,
        #     value_prime=value_prime.detach(),
        #     discount=config.discount,
        # )

        advantage = utils.generalized_advantage_estimation(
            reward_t=rollout.reward,
            value_t=rollout.value.detach(),
            value_prime=value_prime.detach(),
            done_t=rollout.done,
            gamma=config.discount,
            lambda_=0.96,
        )
        value_target = advantage + rollout.value.detach()

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
        agg(loss).backward()
        if config.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(agent.parameters(), config.clip_grad_norm)
        optimizer.step()
        average_reward += config.average_reward_lr * agg(
            td_error.detach()
        )  # TODO: do not use td-error
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

            writer.add_scalar("rollout/average_reward", average_reward, global_step=opt_step)
            grad_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0) for p in agent.parameters()]), 2.0
            )
            writer.add_scalar("rollout/grad_norm", grad_norm, global_step=opt_step)

            for k in [
                "rollout/reward",
                "rollout/value_target",
                "rollout/value",
                "rollout/td_error",
            ]:
                v = metrics[k].compute_and_reset()
                writer.add_scalar(f"{k}/mean", v.mean(), global_step=opt_step)
                writer.add_histogram(f"{k}/hist", v, global_step=opt_step)

            for k in [
                "rollout/entropy",
                "rollout/actor_loss",
                "rollout/critic_loss",
                "rollout/loss",
            ]:
                v = metrics[k].compute_and_reset()
                writer.add_scalar(f"{k}/mean", v.mean(), global_step=opt_step)

            for k in [
                "episode/return",
                "episode/length",
            ]:
                v = metrics[k].compute_and_reset()
                writer.add_scalar(f"{k}/mean", v.mean(), global_step=opt_step)
                writer.add_histogram(f"{k}/hist", v, global_step=opt_step)

            writer.flush()

            # writer.add_scalar(
            #     "rollout/td_error_std_normalized", td_error_std_normalized, global_step=opt_step
            # )

            # torch.save(
            #     {
            #         "agent": agent.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "average_reward": average_reward,
            #     },
            #     "./state.pth",
            # )

    env.close()
    writer.close()


def agg(input):
    assert input.dim() == 2
    # return input.mean(1).sum()
    return input.mean()


def build_env():
    # def scale_reward(r):
    #     return r

    env = "MiniGrid-Empty-Random-6x6-v0"
    # env = "MiniGrid-FourRooms-v0"
    # env = "MiniGrid-FourRooms-Custom-v0"
    # env = "MiniGrid-Dynamic-Obstacles-Random-6x6-v0"
    # env = "MiniGrid-Dynamic-Obstacles-8x8-v0"

    env = gym.make(env)
    env = wrappers.RandomFirstReset(env, 256)
    env = gym_minigrid.wrappers.OneHotPartialObsWrapper(env)
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    # env = gym.wrappers.TransformReward(env, scale_reward)
    # env.reward_range = tuple(map(scale_reward, env.reward_range))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


# def build_env():
#     # env = "MemoryTest-v0"
#     env = "SeqCopy-v0"
#     env = gym.make(env)
#     env = wrappers.RandomFirstReset(env, 32)
#     env = gym.wrappers.RecordEpisodeStatistics(env)
#     return env


def select_action(dist: torch.distributions.Distribution):
    return dist.sample()


if __name__ == "__main__":
    main()
