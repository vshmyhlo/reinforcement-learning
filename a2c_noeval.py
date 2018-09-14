import utils
import numpy as np
import gym
import os
import tensorflow as tf
from tqdm import tqdm
from network import ValueFunction, PolicyCategorical
from vec_env import VecEnv


def build_batch(history):
    columns = zip(*history)

    return [np.array(col).swapaxes(0, 1) for col in columns]


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, default='./tf_log/a2c-noeval')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--entropy-weight', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    utils.fix_seed(args.seed)
    experiment_path = os.path.join(args.experiment_path, args.env)
    env = VecEnv([lambda: gym.make(args.env) for _ in range(os.cpu_count())])
    env.seed(args.seed)

    if args.monitor:
        env = gym.wrappers.Monitor(env, os.path.join('./data', args.env), force=True)

    global_step = tf.train.get_or_create_global_step()
    training = tf.placeholder(tf.bool, [], name='training')

    # input
    b, t = None, None
    states = tf.placeholder(tf.float32, [b, t, *env.observation_space.shape], name='states')
    actions = tf.placeholder(tf.int32, [b, t], name='actions')
    rewards = tf.placeholder(tf.float32, [b, t], name='rewards')
    state_prime = tf.placeholder(tf.float32, [b, *env.observation_space.shape], name='state_prime')
    dones = tf.placeholder(tf.bool, [b, t], name='dones')

    # critic
    value_function = ValueFunction()
    values = value_function(states, training=training)
    value_prime = value_function(state_prime, training=training)
    returns = utils.batch_n_step_return(rewards, value_prime, dones, gamma=args.gamma)
    returns = tf.stop_gradient(returns)
    errors = returns - values
    critic_loss = tf.reduce_mean(tf.square(errors))

    # actor
    policy = PolicyCategorical(np.squeeze(env.action_space.shape))
    dist = policy(states, training=training)
    action_sample = dist.sample()
    advantage = tf.stop_gradient(errors)
    actor_loss = -tf.reduce_mean(dist.log_prob(actions) * advantage)
    actor_loss -= args.entropy_weight * tf.reduce_mean(dist.entropy())

    # training
    loss = actor_loss + critic_loss * 0.5 + tf.losses.get_regularization_loss()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)

    # summary
    ep_length = tf.placeholder(tf.float32, [None])  # TODO: name
    ep_reward = tf.placeholder(tf.float32, [None])  # TODO: name
    metrics, update_metrics = {}, {}
    metrics['loss'], update_metrics['loss'] = tf.metrics.mean(loss)
    metrics['ep_length'], update_metrics['ep_length'] = tf.metrics.mean(ep_length)
    metrics['ep_reward'], update_metrics['ep_reward'] = tf.metrics.mean(ep_reward)
    summary = tf.summary.merge([
        tf.summary.scalar('loss', metrics['loss']),
        tf.summary.scalar('ep_length', metrics['ep_length']),
        tf.summary.scalar('ep_reward', metrics['ep_reward'])
    ])
    locals_init = tf.local_variables_initializer()

    hooks = [
        tf.train.CheckpointSaverHook(checkpoint_dir=experiment_path, save_steps=100)
    ]
    with tf.train.SingularMonitoredSession(checkpoint_dir=experiment_path, hooks=hooks) as sess, tf.summary.FileWriter(
            experiment_path) as writer:
        sess.run(locals_init)
        s = env.reset()
        episode_tracker = utils.EpisodeTracker(s)

        for _ in tqdm(range(args.steps)):
            history = []

            for _ in range(args.horizon):
                a = sess.run(action_sample, {states: np.expand_dims(s, 1)}).squeeze(1)
                s_prime, r, d, _ = env.step(a)
                history.append((s, a, r, d))
                s = np.where(np.expand_dims(d, -1), env.reset(d), s_prime)
                episode_tracker.update(r, d)

            batch = {}
            batch['states'], batch['actions'], batch['rewards'], batch['dones'] = build_batch(history)
            finished_episodes = episode_tracker.reset()

            _, _, step = sess.run(
                [train_step, update_metrics, global_step],
                {
                    states: batch['states'],
                    actions: batch['actions'],
                    rewards: batch['rewards'],
                    state_prime: s_prime,
                    dones: batch['dones'],
                    ep_length: finished_episodes[:, 0],
                    ep_reward: finished_episodes[:, 1]
                })

            if step % 100 == 0:
                step, summ, metr = sess.run([global_step, summary, metrics])
                writer.add_summary(summ, step)
                writer.flush()
                sess.run(locals_init)

    env.close()


if __name__ == '__main__':
    main()
