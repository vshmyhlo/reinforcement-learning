import argparse
import numpy as np
import collections
import gym
import os
import tensorflow as tf
import itertools
from tqdm import tqdm


# TODO: do not mask not taken actions?
# TODO: compute advantage out of graph

class Network(tf.layers.Layer):
    def __init__(self, name='net'):
        super().__init__(name=name)

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.dense_1 = tf.layers.Dense(
            32, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.norm_1 = tf.layers.BatchNormalization()

        self.dense_2 = tf.layers.Dense(
            32, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.norm_2 = tf.layers.BatchNormalization()

    def call(self, input, training):
        input = self.dense_1(input)
        input = self.norm_1(input, training=training)
        input = tf.nn.elu(input)

        input = self.dense_2(input)
        input = self.norm_2(input, training=training)
        input = tf.nn.elu(input)

        return input


class ValueFunction(tf.layers.Layer):
    def __init__(self, name='value_function'):
        super().__init__(name=name)

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.net = Network()
        self.dense = tf.layers.Dense(
            1, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

    def call(self, input, training):
        input = self.net(input, training=training)
        input = self.dense(input)
        input = tf.squeeze(input, -1)

        return input


class Policy(tf.layers.Layer):
    def __init__(self, num_actions, name='policy'):
        super().__init__(name=name)

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.net = Network()
        self.dense = tf.layers.Dense(
            num_actions, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

    def call(self, input, training):
        input = self.net(input, training=training)
        input = self.dense(input)

        dist = tf.distributions.Categorical(input)

        return dist


def sample_history(history, batch_size):
    indices = np.random.permutation(len(history))
    indices = indices[:batch_size]
    batch = [history[i] for i in indices]

    batch = zip(*batch)
    batch = tuple(np.array(x) for x in batch)

    return batch


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    experiment_path = os.path.join(args.experiment_path, args.env)
    env = gym.make(args.env)
    state_size = np.squeeze(env.observation_space.shape)
    assert state_size.shape == ()

    if args.monitor:
        env = gym.wrappers.Monitor(env, os.path.join('./data', args.env), force=True)
    history = collections.deque(maxlen=args.history_size)

    global_step = tf.train.get_or_create_global_step()
    training = tf.placeholder(tf.bool, [])
    value_function = ValueFunction()
    policy = Policy(env.action_space.n)

    # input
    state = tf.placeholder(tf.float32, [None, state_size])
    action = tf.placeholder(tf.int32, [None])
    reward = tf.placeholder(tf.float32, [None])
    state_prime = tf.placeholder(tf.float32, [None, state_size])
    done = tf.placeholder(tf.bool, [None])

    state_value = value_function(state, training=training)
    dist = policy(state, training=training)
    action_sample = dist.sample()
    state_prime_value = value_function(state_prime, training=training)
    td_target = tf.where(done, reward, reward + args.gamma * state_prime_value)

    # actor
    td_error = td_target - state_value
    advantage = tf.stop_gradient(td_error)
    actor_loss = -tf.reduce_mean(dist.log_prob(action) * advantage)
    # actor_loss -= 0.01 * tf.reduce_mean(dist.entropy())

    # critic
    critic_loss = tf.losses.mean_squared_error(
        labels=tf.stop_gradient(td_target),
        predictions=state_value)

    # training
    loss = actor_loss + critic_loss * 0.5 + tf.losses.get_regularization_loss()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)

    metrics, update_metrics = {}, {}
    metrics['loss'], update_metrics['loss'] = tf.metrics.mean(loss)
    metrics['state_value'], update_metrics['state_value'] = tf.metrics.mean(state_value)
    episode_length = tf.placeholder(tf.float32, [])
    episode_reward = tf.placeholder(tf.float32, [])
    summary = tf.summary.merge([
        tf.summary.scalar('loss', metrics['loss']),
        tf.summary.scalar('state_value', metrics['state_value']),
        tf.summary.scalar('episode_length', episode_length),
        tf.summary.scalar('episode_reward', episode_reward)
    ])

    locals_init = tf.local_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess, tf.summary.FileWriter(experiment_path) as writer:
        if tf.train.latest_checkpoint(experiment_path):
            saver.restore(sess, tf.train.latest_checkpoint(experiment_path))
        else:
            sess.run(tf.global_variables_initializer())
        sess.run(locals_init)

        for i in range(args.episodes):
            s = env.reset()
            ep_r = 0

            for t in tqdm(itertools.count(), desc='episode {}, history: {}'.format(i, len(history))):
                a = sess.run(action_sample, {state: s.reshape((1, state_size)), training: False}).squeeze(0)
                s_prime, r, d, _ = env.step(a)
                ep_r += r

                history.append((s, a, r, s_prime, d))
                batch = sample_history(history, args.batch_size)
                _, _, step = sess.run(
                    [train_step, update_metrics, global_step],
                    {
                        state: batch[0],
                        action: batch[1],
                        reward: batch[2],
                        state_prime: batch[3],
                        done: batch[4],
                        training: True
                    })

                if d:
                    break
                else:
                    s = s_prime

            summ, metr = sess.run([summary, metrics], {episode_length: t, episode_reward: ep_r})
            writer.add_summary(summ, step)
            writer.flush()
            saver.save(sess, os.path.join(experiment_path, 'model.ckpt'))
            sess.run(locals_init)


if __name__ == '__main__':
    main()
