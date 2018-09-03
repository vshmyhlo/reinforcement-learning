import utils
import numpy as np
import gym
import os
import tensorflow as tf
import itertools
from tqdm import tqdm
from network import ValueFunction, PolicyCategorical
from vec_env import VecEnv


# TODO: seed env
# TODO: refactor action space differences
# TODO: do not mask not taken actions?
# TODO: compute advantage out of graph
# TODO: test build batch

def build_batch(history, value_prime, gamma):
    state, action, reward, done = [np.array(x).swapaxes(0, 1) for x in zip(*history)]
    ret = utils.batch_a3c_return(reward, value_prime, done, gamma)

    state = utils.flatten_batch_horizon(state)
    action = utils.flatten_batch_horizon(action)
    ret = utils.flatten_batch_horizon(ret)

    return state, action, ret


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=256 // os.cpu_count())
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, default='./tf_log/a2c')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--entropy-weight', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    def train(s, num_steps):
        for _ in tqdm(range(num_steps), desc='training'):
            history = []

            for _ in range(args.horizon):
                a = sess.run(action_sample, {state: s})
                s_prime, r, d, _ = env.step(a)
                history.append((s, a, r, d))
                s = np.where(np.expand_dims(d, -1), env.reset(d), s_prime)

            batch = {}
            v_prime = sess.run(state_value, {state: s})
            batch['state'], batch['action'], batch['return'] = build_batch(history, v_prime, args.gamma)

            sess.run(
                [train_step, update_metrics['loss']],
                {
                    state: batch['state'],
                    action: batch['action'],
                    ret: batch['return']
                })

        return s

    def evaluate(num_episodes):
        env = gym.make(args.env)

        for _ in tqdm(range(num_episodes), desc='evaluating'):
            s = env.reset()
            ep_r = 0

            for t in itertools.count():
                a = sess.run(action_sample, {state: np.expand_dims(s, 0)}).squeeze(0)
                s_prime, r, d, _ = env.step(a)
                ep_r += r

                if d:
                    break
                else:
                    s = s_prime

            sess.run([update_metrics[k] for k in update_metrics if k != 'loss'], {ep_length: t, ep_reward: ep_r})

        step, summ, metr = sess.run([global_step, summary, metrics])
        writer.add_summary(summ, step)
        writer.flush()
        saver.save(sess, os.path.join(experiment_path, 'model.ckpt'))

    args = build_parser().parse_args()
    utils.fix_seed(args.seed)
    experiment_path = os.path.join(args.experiment_path, args.env)
    env = VecEnv([lambda: gym.make(args.env) for _ in range(os.cpu_count())])

    if args.monitor:
        env = gym.wrappers.Monitor(env, os.path.join('./data', args.env), force=True)

    global_step = tf.train.get_or_create_global_step()
    training = tf.placeholder(tf.bool, [], name='training')

    # input
    state = tf.placeholder(tf.float32, [None, *env.observation_space.shape[1:]], name='state')
    action = tf.placeholder(tf.int32, [None], name='action')
    ret = tf.placeholder(tf.float32, [None], name='return')

    # critic
    value_function = ValueFunction()
    state_value = value_function(state, training=training)
    td_error = ret - state_value
    critic_loss = tf.reduce_mean(tf.square(td_error))

    # actor
    policy = PolicyCategorical(env.action_space.n)
    dist = policy(state, training=training)
    action_sample = dist.sample()
    advantage = tf.stop_gradient(td_error)
    actor_loss = -tf.reduce_mean(dist.log_prob(action) * advantage)
    actor_loss -= args.entropy_weight * tf.reduce_mean(dist.entropy())

    # training
    loss = actor_loss + critic_loss * 0.5 + tf.losses.get_regularization_loss()

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
    saver = tf.train.Saver()
    with tf.Session() as sess, tf.summary.FileWriter(experiment_path) as writer:
        if tf.train.latest_checkpoint(experiment_path):
            saver.restore(sess, tf.train.latest_checkpoint(experiment_path))
        else:
            sess.run(tf.global_variables_initializer())

        s = env.reset()
        for _ in range(args.steps // 100):
            sess.run(locals_init)
            s = train(s, num_steps=100)
            evaluate(num_episodes=10)

    env.close()


if __name__ == '__main__':
    main()
