import utils
import ray
import numpy as np
import gym
import os
import tensorflow as tf
import itertools
from network import ValueFunction, PolicyCategorical


# TODO: refactor action space differences
# TODO: do not mask not taken actions?
# TODO: compute advantage out of graph
# TODO: test build batch

def build_batch(history, gamma):
    s, a, r = zip(*history)
    r = utils.discounted_return(np.array(r), gamma)

    return s, a, r


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--history-size', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, default='./tf_log/ac-mc-distributed')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--monitor', action='store_true')

    return parser


@ray.remote
class Master(object):
    def __init__(self, config):
        experiment_path = os.path.join(config.experiment_path, config.env)
        env = gym.make(config.env)
        state_size = np.squeeze(env.observation_space.shape)
        assert state_size.shape == ()

        self._global_step = tf.train.get_or_create_global_step()
        training = tf.placeholder(tf.bool, [], name='training')

        # input
        state = tf.placeholder(tf.float32, [None, state_size], name='state')

        # critic
        value_function = ValueFunction()
        value_function(state, training=training)

        # actor
        policy = PolicyCategorical(env.action_space.n)
        policy(state, training=training)

        # training
        opt = tf.train.AdamOptimizer(config.learning_rate)
        self._vars = tf.trainable_variables()
        self._grads = [tf.placeholder(var.dtype, var.shape) for var in self._vars]
        grads_and_vars = zip(self._grads, self._vars)
        self._apply_gradients = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

        # summary
        self._ep_length = tf.placeholder(tf.float32, [])
        self._ep_reward = tf.placeholder(tf.float32, [])
        self._metrics, self._update_metrics = {}, {}
        self._metrics['ep_length'], self._update_metrics['ep_length'] = tf.metrics.mean(self._ep_length)
        self._metrics['ep_reward'], self._update_metrics['ep_reward'] = tf.metrics.mean(self._ep_reward)
        self._summary = tf.summary.merge([
            tf.summary.scalar('ep_length', self._metrics['ep_length']),
            tf.summary.scalar('ep_reward', self._metrics['ep_reward'])
        ])
        self._locals_init = tf.local_variables_initializer()

        # session
        hooks = [
            tf.train.CheckpointSaverHook(checkpoint_dir=experiment_path, save_steps=100)
        ]
        self._sess = tf.train.SingularMonitoredSession(checkpoint_dir=experiment_path, hooks=hooks)
        self._writer = tf.summary.FileWriter(experiment_path)

    def sync(self, gs, t, total_reward):
        _, _, step = self._sess.run(
            [self._apply_gradients, self._update_metrics, self._global_step],
            {
                self._ep_length: t,
                self._ep_reward: total_reward,
                **{grad: g for grad, g in zip(self._grads, gs)}
            })
        updates = self._sess.run(self._vars)

        if step % 100 == 0:
            summ, metr = self._sess.run([self._summary, self._metrics])
            self._writer.add_summary(summ, step)
            self._writer.flush()
            self._sess.run(self._locals_init)
            print(metr)

        return updates

    def updates(self):
        updates = self._sess.run(self._vars)

        return updates


@ray.remote
class Worker(object):
    def __init__(self, master, config):
        self._master = master
        self._config = config

    def train(self):
        env = gym.make(self._config.env)
        state_size = np.squeeze(env.observation_space.shape)
        assert state_size.shape == ()

        # if args.monitor:
        #     env = gym.wrappers.Monitor(env, os.path.join('./data', args.env), force=True)

        training = tf.placeholder(tf.bool, [], name='training')

        # input
        state = tf.placeholder(tf.float32, [None, state_size], name='state')
        action = tf.placeholder(tf.int32, [None], name='action')
        ret = tf.placeholder(tf.float32, [None], name='return')

        # critic
        value_function = ValueFunction()
        state_value = value_function(state, training=training)
        td_target = tf.stop_gradient(ret)
        td_error = td_target - state_value
        critic_loss = tf.reduce_mean(tf.square(td_error))

        # actor
        policy = PolicyCategorical(env.action_space.n)
        dist = policy(state, training=training)
        action_sample = dist.sample()
        advantage = tf.stop_gradient(td_error)
        actor_loss = -tf.reduce_mean(dist.log_prob(action) * advantage)
        actor_loss -= 1e-3 * tf.reduce_mean(dist.entropy())

        # training
        loss = actor_loss + critic_loss * 0.5 + tf.losses.get_regularization_loss()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(self._config.learning_rate)
            grads_and_vars = opt.compute_gradients(loss)
            grads, vars = zip(*grads_and_vars)

        updates = [tf.placeholder(var.dtype, var.shape) for var in vars]
        apply_updates = tf.group(*[var.assign(update) for var, update in zip(vars, updates)])

        with tf.Session() as sess:
            us = ray.get(self._master.updates.remote())
            sess.run(apply_updates, {update: u for update, u in zip(updates, us)})

            for ep in range(self._config.episodes):
                history = []
                s = env.reset()
                total_reward = 0

                for t in itertools.count():
                    a = sess.run(action_sample, {state: s.reshape((1, state_size)), training: False}).squeeze(0)
                    s_prime, r, d, _ = env.step(a)
                    total_reward += r

                    history.append((s, a, r))

                    if d:
                        break
                    else:
                        s = s_prime

                batch = build_batch(history, self._config.gamma)

                gs = sess.run(grads, {
                    state: batch[0],
                    action: batch[1],
                    ret: batch[2],
                    training: True,
                })
                us = ray.get(self._master.sync.remote(gs, t, total_reward))
                sess.run(apply_updates, {update: u for update, u in zip(updates, us)})


def main():
    args = build_parser().parse_args()
    # utils.fix_seed(args.seed)
    ray.init()
    master = Master.remote(args)
    workers = [Worker.remote(master, args) for _ in range(os.cpu_count())]
    tasks = [w.train.remote() for w in workers]
    ray.get(tasks)


if __name__ == '__main__':
    main()
