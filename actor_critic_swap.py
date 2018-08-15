import argparse
import collections
import numpy as np
import gym
import os
import tensorflow as tf
import itertools
from tqdm import tqdm
from network import ValueFunction, PolicyCategorical


def sample_history(history, batch_size):
    indices = np.random.permutation(len(history))
    indices = indices[:batch_size]
    batch = [history[i] for i in indices]

    batch = zip(*batch)
    batch = tuple(np.array(x) for x in batch)

    return batch


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-size', type=int, default=50000)
    parser.add_argument('--value-update-interval', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
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

    global_step = tf.train.get_or_create_global_step()
    training = tf.placeholder(tf.bool, [])

    # input
    state = tf.placeholder(tf.float32, [None, state_size])
    action = tf.placeholder(tf.int32, [None])
    reward = tf.placeholder(tf.float32, [None])
    state_prime = tf.placeholder(tf.float32, [None, state_size])
    done = tf.placeholder(tf.bool, [None])

    # critic
    value_function = ValueFunction(name='value_function')
    state_value = value_function(state, training=training)

    value_function_old = ValueFunction(trainable=False, name='value_function_old')
    state_prime_value_old = value_function_old(state_prime, training=training)
    td_target_old = tf.where(done, reward, reward + args.gamma * state_prime_value_old)

    critic_loss = tf.losses.mean_squared_error(
        labels=tf.stop_gradient(td_target_old),
        predictions=state_value)

    # actor
    policy = PolicyCategorical(env.action_space.n)
    dist = policy(state, training=training)
    action_sample = dist.sample()
    td_error = td_target_old - state_value
    advantage = tf.stop_gradient(td_error)
    actor_loss = -tf.reduce_mean(dist.log_prob(action) * advantage)
    actor_loss -= 1e-3 * tf.reduce_mean(dist.entropy())

    # training
    loss = actor_loss + critic_loss * 0.5 + tf.losses.get_regularization_loss()

    value_function_vars = tf.global_variables('value_function/')
    value_function_old_vars = tf.global_variables('value_function_old/')
    assert len(value_function_vars) == len(value_function_old_vars)
    value_function_old_update = tf.group(*[
        var_old.assign(var) for var, var_old in zip(value_function_vars, value_function_old_vars)])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)

    metrics, update_metrics = {}, {}
    metrics['loss'], update_metrics['loss'] = tf.metrics.mean(loss)
    episode_length = tf.placeholder(tf.float32, [])
    episode_reward = tf.placeholder(tf.float32, [])
    summary = tf.summary.merge([
        tf.summary.scalar('loss', metrics['loss']),
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
            sess.run(value_function_old_update)

        history = collections.deque(maxlen=args.history_size)

        s = env.reset()
        for _ in tqdm(range(args.history_size // 10), desc='building history'):
            a = env.action_space.sample()
            s_prime, r, d, _ = env.step(a)
            history.append((s, a, r, s_prime, d))

            if d:
                s = env.reset()
            else:
                s = s_prime

        assert len(history) == args.history_size // 10

        for i in range(args.episodes):
            sess.run(locals_init)
            s = env.reset()
            ep_r = 0

            for t in tqdm(itertools.count(), desc='episode {}, history size {}'.format(i, len(history))):
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

                if step % args.value_update_interval == 0:
                    sess.run(value_function_old_update)

                if d:
                    break
                else:
                    s = s_prime

            summ, metr = sess.run([summary, metrics], {episode_length: t, episode_reward: ep_r})
            writer.add_summary(summ, step)
            writer.flush()
            saver.save(sess, os.path.join(experiment_path, 'model.ckpt'))


if __name__ == '__main__':
    main()
