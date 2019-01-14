import utils
import numpy as np
import gym
import os
import tensorflow as tf
import itertools
from tqdm import tqdm
from network import PolicyCategorical


def build_batch(history):
    columns = zip(*history)

    return [np.array(col).swapaxes(0, 1) for col in columns]


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, default='./tf_log/pg-mc')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--entropy-weight', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    utils.fix_seed(args.seed)
    experiment_path = os.path.join(args.experiment_path, args.env)
    env = gym.make(args.env)
    env.seed(args.seed)

    if args.monitor:
        env = gym.wrappers.Monitor(env, os.path.join('./data', args.env), force=True)

    global_step = tf.train.get_or_create_global_step()
    training = tf.placeholder(tf.bool, [], name='training')

    # input
    b, t = 1, None
    states = tf.placeholder(tf.float32, [b, t, *env.observation_space.shape], name='states')
    actions = tf.placeholder(tf.int32, [b, t], name='actions')
    rewards = tf.placeholder(tf.float32, [b, t], name='rewards')

    # actor
    policy = PolicyCategorical(np.squeeze(env.action_space.shape))
    dist = policy(states, training=training)
    action_sample = dist.sample()
    returns = utils.batch_return(rewards, gamma=args.gamma)
    advantages = tf.stop_gradient(returns)  # TODO: normalize?
    actor_loss = -tf.reduce_mean(dist.log_prob(actions) * advantages)
    actor_loss -= args.entropy_weight * tf.reduce_mean(dist.entropy())

    # training
    loss = actor_loss + tf.losses.get_regularization_loss()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)

    # summary
    ep_length = tf.placeholder(tf.float32, [])
    ep_reward = tf.placeholder(tf.float32, [])
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

        for ep in tqdm(range(args.episodes), desc='training'):
            history = []
            s = env.reset()
            ep_r = 0

            for t in itertools.count():
                a = sess.run(action_sample, {states: np.reshape(s, (1, 1, -1))}).squeeze((0, 1))
                s_prime, r, d, _ = env.step(a)
                ep_r += r

                history.append(([s], [a], [r]))

                if d:
                    break
                else:
                    s = s_prime

            batch = {}
            batch['states'], batch['actions'], batch['rewards'] = build_batch(history)

            _, _, step = sess.run(
                [train_step, update_metrics, global_step],
                {
                    states: batch['states'],
                    actions: batch['actions'],
                    rewards: batch['rewards'],
                    ep_length: t,
                    ep_reward: ep_r
                })

            if ep % 100 == 0:
                summ, metr = sess.run([summary, metrics])
                writer.add_summary(summ, step)
                writer.flush()
                sess.run(locals_init)


if __name__ == '__main__':
    main()
