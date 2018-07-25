import argparse
import gym
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

    return parser


def main():
    args = build_parser().parse_args()

    q_function = Q(num_actions=6)

    observation = tf.placeholder(tf.uint8, [args.num_last_states, 210, 160, 3])
    state = preprocess(observation)
    state_value = q_function(tf.expand_dims(state, 0), training=False)
    e_greedy_action = epsilon_greedy(state_value)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        env = LastNStatesEnvironment(gym.make('SpaceInvaders-v0'), num_last_states=args.num_last_states)
        obs = env.reset()
        s, a = sess.run([state, e_greedy_action], {observation: obs})

        # build_history
        history = []
        for _ in tqdm(range(args.history_size)):
            obs_prime, r, done, _ = env.step(a.squeeze())
            s_prime, a_prime = sess.run([state, e_greedy_action], {observation: obs_prime})
            history.append((s.squeeze(), a.squeeze(), r, s_prime.squeeze()))

            if done:
                obs = env.reset()
                s, a = sess.run([state, e_greedy_action], {observation: [obs]})
            else:
                s, a = s_prime, a_prime

        # train
        gamma = tf.placeholder(tf.float32, [])

        state = tf.placeholder(tf.float32, [None, None, None, args.num_last_states])
        action = tf.placeholder(tf.int64, [None])
        reward = tf.placeholder(tf.float32, [None])
        state_prime = tf.placeholder(tf.float32, [None, None, None, args.num_last_states])

        state_value = q_function(state, training=True)  # TODO: training
        action_value_pred = utils.select_action_value(state_value, action)

        state_value_prime = q_function(state_prime, training=False)  # TODO: training
        action_value_prime = utils.select_action_value(state_value_prime, greedy(state_value_prime))
        action_value_true = reward + gamma * action_value_prime  # TODO: rename

        loss = mse_loss(labels=tf.stop_gradient(action_value_true), logits=action_value_pred)  # TODO: stop grad
        train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

        for s, a, r, s_prime in tqdm(batched_history(history, args.batch_size)):  # TODO: shuffle
            print()
            print(s.shape)
            print(a.shape)
            print(r.shape)
            print(s_prime.shape)
            print()

            sess.run(train_step, {state: s, action: a, reward: r, state_prime: s_prime, gamma: 0.9})


if __name__ == '__main__':
    main()
