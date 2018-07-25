import logging
import argparse
import gym
import os
import numpy as np
import itertools
import tensorflow as tf
from tqdm import tqdm
from q_function import Q
import utils
import collections


# TODO: rename action values to states value
# TODO: check supor().__init__() and super().build() order and availability
# TODO: check training arg

def mse_loss(labels, logits, name='mse_loss'):
    with tf.name_scope(name):
        loss = tf.square(labels - logits)
        loss = tf.reduce_mean(loss)

        return loss


def greedy(state_value):
    return tf.argmax(state_value, -1)


def epsilon_greedy(state_value, eps=0.1):  # TODO: epsilon anealing
    p = tf.random_uniform([])
    shape = tf.shape(state_value)

    return tf.cond(
        p > eps,
        lambda: greedy(state_value),
        lambda: tf.random_uniform([shape[0]], 0, tf.to_int64(shape[1]), dtype=tf.int64))


def preprocess(observation):
    observation = tf.image.convert_image_dtype(observation, tf.float32)
    observation = tf.image.rgb_to_grayscale(observation)
    observation = tf.transpose(observation, [3, 1, 2, 0])
    observation = tf.squeeze(observation, 0)

    return observation


# TODO: shuffle

def batched_history(history, batch_size):
    history = iter(history)

    while True:
        batch = list(itertools.islice(history, batch_size))
        size = len(batch)
        batch = tuple(zip(*batch))
        batch = tuple(np.array(x) for x in batch)

        yield batch

        if size < batch_size:
            break


class LastNStatesEnvironment(object):
    def __init__(self, env, num_last_states):
        self._env = env
        self._num_last_states = num_last_states

    def reset(self):
        self._queue = collections.deque(maxlen=self._num_last_states)
        obs = self._env.reset()
        self._queue.append(obs)

        while len(self._queue) < self._num_last_states:
            obs, _, _, _ = self._env.step(self._env.action_space.sample())
            self._queue.append(obs)

        obs = np.stack(self._queue, 0)
        return obs

    def step(self, action):
        obs, rew, done, meta = self._env.step(action)
        self._queue.append(obs)
        obs = np.stack(self._queue, 0)
        return obs, rew, done, meta


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-size', type=int, default=2000)
    parser.add_argument('--num-last-states', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--experiment-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1000)

    return parser


class HistoryBuilder(object):
    def __init__(self, q_function, num_last_states):
        self._observation = tf.placeholder(tf.uint8, [num_last_states, 210, 160, 3])
        self._state = preprocess(self._observation)
        state_value = q_function(tf.expand_dims(self._state, 0), training=False)
        self._e_greedy_action = epsilon_greedy(state_value)

    def __call__(self, env, obs, sess, history_size):
        s, a = sess.run([self._state, self._e_greedy_action], {self._observation: obs})

        history = []
        for _ in tqdm(range(history_size)):
            obs_prime, r, done, _ = env.step(a.squeeze())
            s_prime, a_prime = sess.run([self._state, self._e_greedy_action], {self._observation: obs_prime})
            history.append((s.squeeze(), a.squeeze(), r, s_prime.squeeze()))

            if done:
                obs = env.reset()
                s, a = sess.run([self._state, self._e_greedy_action], {self._observation: [obs]})
            else:
                s, a = s_prime, a_prime

        return history


class Trainer(object):
    def __init__(self, q_function, num_last_states):
        global_step = tf.train.get_or_create_global_step()

        gamma = 0.9  # TODO: find gammma

        self._state = tf.placeholder(tf.float32, [None, None, None, num_last_states])
        self._action = tf.placeholder(tf.int64, [None])
        self._reward = tf.placeholder(tf.float32, [None])
        self._state_prime = tf.placeholder(tf.float32, [None, None, None, num_last_states])

        state_value = q_function(self._state, training=True)  # TODO: training
        action_value_pred = utils.select_action_value(state_value, self._action)

        state_value_prime = q_function(self._state_prime, training=False)  # TODO: training
        action_value_prime = utils.select_action_value(state_value_prime, greedy(state_value_prime))
        action_value_true = self._reward + gamma * action_value_prime  # TODO: rename

        loss = mse_loss(labels=tf.stop_gradient(action_value_true), logits=action_value_pred)  # TODO: stop grad
        self._train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss, global_step=global_step)

    def __call__(self, history, sess):
        epochs = 5  # TODO: num epochs

        for epoch in range(epochs):
            for s, a, r, s_prime in tqdm(history):  # TODO: shuffle
                sess.run(
                    self._train_step, {self._state: s, self._action: a, self._reward: r, self._state_prime: s_prime})


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()

    q_function = Q(num_actions=6)
    history_build = HistoryBuilder(q_function, num_last_states=args.num_last_states)
    trainer = Trainer(q_function, num_last_states=args.num_last_states)

    saver = tf.train.Saver()
    with tf.Session() as sess, tf.summary.FileWriter(args.experiment_path) as writer:
        if tf.train.latest_checkpoint(args.experiment_path):
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, tf.train.latest_checkpoint(args.experiment_path))

        # init
        env = LastNStatesEnvironment(gym.make('SpaceInvaders-v0'), num_last_states=args.num_last_states)
        obs = env.reset()

        for epoch in range(args.epochs):
            logging.info('epoch {}'.format(epoch))

            # build history
            history = history_build(env, obs, sess=sess, history_size=args.history_size)

            # train
            trainer(batched_history(history, args.batch_size), sess=sess)

            # save
            saver.save(sess, os.path.join(args.experiment_path, 'model.ckpt'))


if __name__ == '__main__':
    main()
