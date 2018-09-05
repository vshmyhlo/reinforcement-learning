import utils
from vec_env import VecEnv
import numpy as np
import gym
import os
import tensorflow as tf
import itertools
from tqdm import tqdm
from network import PolicyCategorical, ValueFunction


# TODO: multiepoch
# TODO: do not mask not taken actions?
# TODO: compute advantage out of graph


def history_to_batch(history):
    columns = zip(*history)

    return [np.array(col).swapaxes(0, 1) for col in columns]


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--history-size', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, default='./tf_log/ppo')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae', type=float, default=0.95)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--entropy-weight', type=float, default=1e-2)
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    def train(s, num_steps):
        for _ in tqdm(range(num_steps), desc='training'):
            history = []

            for _ in range(args.horizon):
                a, v = sess.run([action_sample, state_value], {state: s})
                s_prime, r, d, _ = env.step(a)
                history.append((s, a, r, v, d))
                s = np.where(np.expand_dims(d, -1), env.reset(d), s_prime)

            batch = {}
            batch['state'], batch['action'], batch['reward'], batch['value'], batch['done'] = history_to_batch(history)
            batch['value_prime'] = sess.run(state_value, {state: s_prime})
            batch['advantage'] = utils.generalized_advantage_estimation(
                batch['reward'], batch['value'], batch['value_prime'], batch['done'], gamma=args.gamma, lam=args.gae)
            batch['advantage'] = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-5)
            batch['return'] = batch['advantage'] + batch['value']

            sess.run(
                [train_step, update_metrics['loss']],
                {
                    state: utils.flatten_batch_horizon(batch['state']),
                    action: utils.flatten_batch_horizon(batch['action']),
                    ret: utils.flatten_batch_horizon(batch['return']),
                    advantage: utils.flatten_batch_horizon(batch['advantage'])
                })

            sess.run(update_policy_old)

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
    advantage = tf.placeholder(tf.float32, [None], name='advantage')

    # critic
    value_function = ValueFunction()
    state_value = value_function(state, training=training)
    critic_loss = tf.reduce_mean(tf.square(ret - state_value))

    # actor
    policy = PolicyCategorical(env.action_space.n, name='policy')
    dist = policy(state, training=training)
    policy_old = PolicyCategorical(env.action_space.n, trainable=False, name='policy_old')
    dist_old = policy_old(state, training=False)
    action_sample = dist.sample()

    ratio = tf.exp(dist.log_prob(action) - dist_old.log_prob(action))  # pnew / pold
    surr1 = ratio * advantage  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage  #
    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
    actor_loss -= args.entropy_weight * tf.reduce_mean(dist.entropy())

    update_policy_old = tf.group(*[
        tf.assign(old_var, var)
        for var, old_var in zip(tf.global_variables('policy/'), tf.global_variables('policy_old/'))])

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

    hooks = [
        tf.train.CheckpointSaverHook(checkpoint_dir=experiment_path, save_steps=100)
    ]
    with tf.train.SingularMonitoredSession(checkpoint_dir=experiment_path, hooks=hooks) as sess, tf.summary.FileWriter(
            experiment_path) as writer:
        s = env.reset()
        for _ in range(1000):
            sess.run(locals_init)
            s = train(s, num_steps=100)
            evaluate(num_episodes=10)

    env.close()


if __name__ == '__main__':
    main()
