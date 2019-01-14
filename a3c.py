import tensorflow as tf
from tqdm import tqdm
import numpy as np
import gym
import os
import itertools
import ray
from network import ValueFunction, PolicyCategorical
import utils


# TODO: num steps
# TODO: track loss
# TODO: seed
# TODO: shared rms

def build_batch(history):
    columns = zip(*history)

    return [np.array(col).swapaxes(0, 1) for col in columns]


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, default='./tf_log/a3c')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--entropy-weight', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--monitor', action='store_true')

    return parser


@ray.remote
class Master(object):
    def __init__(self, args):
        experiment_path = os.path.join(args.experiment_path, args.env)
        env = gym.make(args.env)
        self.global_step = tf.train.get_or_create_global_step()  # TODO: self?
        training = True

        # input
        b, t = 1, None
        states = tf.placeholder(tf.float32, [b, t, *env.observation_space.shape], name='states')

        # critic
        value_function = ValueFunction()
        value_function(states, training=training)

        # actor
        policy = PolicyCategorical(np.squeeze(env.action_space.shape))
        policy(states, training=training)

        # training
        self.vars = tf.trainable_variables()
        self.grad_holders = [tf.placeholder(var.dtype, var.shape) for var in self.vars]
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        grads_and_vars = zip(self.grad_holders, self.vars)
        self.apply_gradients = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # summary
        self.ep_length = tf.placeholder(tf.float32, [])
        self.ep_reward = tf.placeholder(tf.float32, [])
        metrics, self.update_metrics = {}, {}
        metrics['ep_length'], self.update_metrics['ep_length'] = tf.metrics.mean(self.ep_length)
        metrics['ep_reward'], self.update_metrics['ep_reward'] = tf.metrics.mean(self.ep_reward)
        self.summary = tf.summary.merge([
            tf.summary.scalar('ep_length', metrics['ep_length']),
            tf.summary.scalar('ep_reward', metrics['ep_reward'])
        ])

        # init
        self.locals_init = tf.local_variables_initializer()

        # session
        hooks = [
            tf.train.CheckpointSaverHook(checkpoint_dir=experiment_path, save_steps=100)
        ]
        self.sess = tf.train.SingularMonitoredSession(checkpoint_dir=experiment_path, hooks=hooks)
        self.writer = tf.summary.FileWriter(experiment_path)
        self.sess.run(self.locals_init)
        self.tqdm = tqdm()

    def update(self, gs):
        _, vs, step = self.sess.run(
            [self.apply_gradients, self.vars, self.global_step],
            {grad: g for grad, g in zip(self.grad_holders, gs)})

        self.tqdm.update()

        if step % 100 == 0:
            summ = self.sess.run(self.summary)
            self.writer.add_summary(summ, step)
            self.writer.flush()
            self.sess.run(self.locals_init)

        return vs

    def metrics(self, ep_len, ep_rew):
        self.sess.run(self.update_metrics, {self.ep_length: ep_len, self.ep_reward: ep_rew})


@ray.remote
class Worker(object):
    def __init__(self, master, args):
        self.master = master
        self.args = args
        self.env = gym.make(args.env)
        training = True

        # input
        b, t = 1, None
        self.states = tf.placeholder(tf.float32, [b, t, *self.env.observation_space.shape], name='states')
        self.actions = tf.placeholder(tf.int32, [b, t], name='actions')
        self.rewards = tf.placeholder(tf.float32, [b, t], name='rewards')
        self.state_prime = tf.placeholder(tf.float32, [b, *self.env.observation_space.shape], name='state_prime')
        self.dones = tf.placeholder(tf.bool, [b, t], name='dones')

        # critic
        value_function = ValueFunction()
        values = value_function(self.states, training=training)
        value_prime = value_function(self.state_prime, training=training)
        returns = utils.batch_n_step_return(self.rewards, value_prime, self.dones, gamma=args.gamma)
        returns = tf.stop_gradient(returns)
        errors = returns - values
        critic_loss = tf.reduce_mean(tf.square(errors))

        # actor
        policy = PolicyCategorical(np.squeeze(self.env.action_space.shape))
        dist = policy(self.states, training=training)
        self.action_sample = dist.sample()
        advantages = tf.stop_gradient(errors)  # TODO: normalization
        actor_loss = -tf.reduce_mean(dist.log_prob(self.actions) * advantages)
        actor_loss -= args.entropy_weight * tf.reduce_mean(dist.entropy())

        # training
        loss = actor_loss + critic_loss * 0.5 + tf.losses.get_regularization_loss()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)

        self.grads, vars = zip(*grads_and_vars)
        self.var_holders = [tf.placeholder(var.dtype, var.shape) for var in vars]
        self.update_vars = tf.group(*[var.assign(var_holder) for var, var_holder in zip(vars, self.var_holders)])

        # init
        self.globals_init = tf.global_variables_initializer()

        # session
        self.sess = tf.Session()
        self.sess.run(self.globals_init)

    def train(self):
        s = self.env.reset()
        t = 0
        ep_rew = 0

        for _ in itertools.count():
            history = []
            for _ in range(self.args.horizon):
                a = self.sess.run(self.action_sample, {self.states: np.reshape(s, (1, 1, -1))}).squeeze((0, 1))
                s_prime, r, d, _ = self.env.step(a)

                t += 1
                ep_rew += r

                history.append(([s], [a], [r], [d]))

                if d:
                    self.master.metrics.remote(t, ep_rew)

                    s = self.env.reset()
                    t = 0
                    ep_rew = 0

                    break
                else:
                    s = s_prime

            batch = {}
            batch['states'], batch['actions'], batch['rewards'], batch['dones'] = build_batch(history)

            gs = self.sess.run(
                self.grads,
                {

                    self.states: batch['states'],
                    self.actions: batch['actions'],
                    self.rewards: batch['rewards'],
                    self.state_prime: [s_prime],
                    self.dones: batch['dones']
                })
            vs = ray.get(self.master.update.remote(gs))
            self.sess.run(self.update_vars, {var_holder: v for var_holder, v in zip(self.var_holders, vs)})


def main():
    args = build_parser().parse_args()

    ray.init()

    master = Master.remote(args)
    workers = [Worker.remote(master, args) for _ in range(args.workers)]
    tasks = [worker.train.remote() for worker in workers]
    ray.get(tasks)


if __name__ == '__main__':
    main()
