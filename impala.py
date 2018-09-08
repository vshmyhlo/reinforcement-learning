import os
from network import ValueFunction, PolicyCategorical
import itertools
import tensorflow as tf
import utils
import gym
import numpy as np
import ray


# TODO: plural names
# TODO: join value function and policy everywhere
# TODO: refactor net creation
# TODO: use args.episodes
# TODO: print ratio

def from_importance_weights(
        log_ratios, discounts, rewards, values, value_prime, clip_ratio_threshold=1.0, clip_pg_ratio_threshold=1.0,
        name='from_importance_weights'):
    with tf.name_scope(name):
        log_ratios = tf.convert_to_tensor(log_ratios, dtype=tf.float32)
        discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        values = tf.convert_to_tensor(values, dtype=tf.float32)
        value_prime = tf.convert_to_tensor(value_prime, dtype=tf.float32)
        if clip_ratio_threshold is not None:
            clip_ratio_threshold = tf.convert_to_tensor(clip_ratio_threshold, dtype=tf.float32)
        if clip_pg_ratio_threshold is not None:
            clip_pg_ratio_threshold = tf.convert_to_tensor(clip_pg_ratio_threshold, dtype=tf.float32)

        ratios = tf.exp(log_ratios)
        if clip_ratio_threshold is not None:
            clipped_ratios = tf.minimum(clip_ratio_threshold, ratios)
        else:
            clipped_ratios = ratios

        cs = tf.minimum(1.0, ratios)  # TODO: why cs is computed like this?
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_prime = tf.concat([values[1:], tf.expand_dims(value_prime, 0)], axis=0)
        deltas = clipped_ratios * (rewards + discounts * values_prime - values)

        # V-trace vs are calculated through a scan from the back to the beginning
        # of the given trajectory.
        def scanfunc(acc, sequence_item):
            discount_t, c_t, delta_t = sequence_item
            return delta_t + discount_t * c_t * acc

        initial_values = tf.zeros_like(value_prime)
        vs_minus_values = tf.scan(
            fn=scanfunc,
            elems=(discounts, cs, deltas),
            initializer=initial_values,
            parallel_iterations=1,
            back_prop=False,
            reverse=True)

        # Add V(x_s) to get v_s.
        vs = vs_minus_values + values

        # Advantage for policy gradient.
        vs_prime = tf.concat([vs[1:], tf.expand_dims(value_prime, 0)], axis=0)
        if clip_pg_ratio_threshold is not None:
            clipped_pg_ratios = tf.minimum(clip_pg_ratio_threshold, ratios)
        else:
            clipped_pg_ratios = ratios
        pg_advantages = clipped_pg_ratios * (rewards + discounts * vs_prime - values)

        return tf.stop_gradient(vs), tf.stop_gradient(pg_advantages)


class ValueFunctionAndPolicy(tf.layers.Layer):
    def __init__(self, num_actions, name='value_function_and_policy'):
        super().__init__(name=name)

        self.value_function = ValueFunction()
        self.policy = PolicyCategorical(num_actions)

    def call(self, input, training):
        value = self.value_function(input, training=training)
        dist = self.policy(input, training=training)

        return value, dist


class Master(object):
    def __init__(self, config):
        self.config = config
        env = gym.make(config.env)
        state_size = np.squeeze(env.observation_space.shape)
        assert state_size.shape == ()

        experiment_path = os.path.join(config.experiment_path, config.env)

        self.global_step = tf.train.get_or_create_global_step()
        training = tf.placeholder(tf.bool, [], name='training')

        # input
        t = config.horizon
        n = config.batch_size
        self.states = tf.placeholder(tf.float32, [t, n, state_size], name='state')
        self.state_prime = tf.placeholder(tf.float32, [n, state_size], name='state_prime')
        self.actions = tf.placeholder(tf.int32, [t, n], name='action')
        self.rewards = tf.placeholder(tf.float32, [t, n], name='return')
        self.dones = tf.placeholder(tf.bool, [t, n], name='done')

        # network
        value_function_and_policy = ValueFunctionAndPolicy(np.squeeze(env.action_space.shape))
        values, dist = value_function_and_policy(self.states, training=training)
        value_prime, _ = value_function_and_policy(self.state_prime, training=training)

        # v-trace
        target_actions_prob = dist.prob(self.actions)
        self.behaviour_actions_prob = tf.placeholder(tf.float32, [None, None])
        log_ratios = tf.log(target_actions_prob / self.behaviour_actions_prob)
        discounts = tf.ones_like(log_ratios) * config.gamma * tf.to_float(~self.dones)
        td_target, advantage = from_importance_weights(
            log_ratios=log_ratios, discounts=discounts, rewards=self.rewards, values=values, value_prime=value_prime)

        # critic
        td_error = td_target - values
        critic_loss = tf.reduce_mean(tf.square(td_error))

        # actor
        actor_loss = -tf.reduce_mean(dist.log_prob(self.actions) * advantage)
        actor_loss -= 1e-3 * tf.reduce_mean(dist.entropy())

        # training
        loss = actor_loss + critic_loss * 0.5 + tf.losses.get_regularization_loss()
        self.vars = tf.trainable_variables()
        self.train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss, global_step=self.global_step)

        # summary
        self.ep_length = tf.placeholder(tf.float32, [])
        self.ep_reward = tf.placeholder(tf.float32, [])
        self.metrics, self.update_metrics = {}, {}
        self.metrics['loss'], self.update_metrics['loss'] = tf.metrics.mean(loss)
        self.metrics['ep_length'], self.update_metrics['ep_length'] = tf.metrics.mean(self.ep_length)
        self.metrics['ep_reward'], self.update_metrics['ep_reward'] = tf.metrics.mean(self.ep_reward)
        summary = tf.summary.merge([
            tf.summary.scalar('ep_length', self.metrics['ep_length']),
            tf.summary.scalar('ep_reward', self.metrics['ep_reward'])
        ])
        self.locals_init = tf.local_variables_initializer()

        # session
        hooks = [
            tf.train.CheckpointSaverHook(checkpoint_dir=experiment_path, save_steps=100),
            tf.train.SummarySaverHook(output_dir=experiment_path, save_steps=100, summary_op=summary)
        ]
        self.sess = tf.train.SingularMonitoredSession(checkpoint_dir=experiment_path, hooks=hooks)
        self.history = []

    def updates(self):
        return self.sess.run(self.vars)

    def episode_metrics(self, t, total_reward):
        self.sess.run(
            [self.update_metrics['ep_length'], self.update_metrics['ep_reward']], {
                self.ep_length: t,
                self.ep_reward: total_reward
            })

    def train(self, batch):
        self.history.append(batch)

        if len(self.history) < self.config.batch_size:
            return self.sess.run(self.vars)

        batch = zip(*self.history)
        states, actions, actions_prob, rewards, state_prime, dones = batch
        states, actions, actions_prob, rewards, dones = [
            np.stack(x, 1) for x in [states, actions, actions_prob, rewards, dones]]
        state_prime = np.stack(state_prime, 0)
        self.history = []

        _, _, vs, step = self.sess.run(
            [self.train_step, self.update_metrics['loss'], self.vars, self.global_step],
            {
                self.states: states,
                self.state_prime: state_prime,
                self.actions: actions,
                self.behaviour_actions_prob: actions_prob,
                self.rewards: rewards,
                self.dones: dones
            }
        )

        if step % 100 == 0:
            print(step)
            self.sess.run(self.locals_init)

        return vs


class Worker(object):
    def __init__(self, master, config):
        self.master = master
        self.config = config

    def train(self):
        env = gym.make(self.config.env)
        state_size = np.squeeze(env.observation_space.shape)
        assert state_size.shape == ()

        training = tf.placeholder(tf.bool, [], name='training')

        # input
        state = tf.placeholder(tf.float32, [None, state_size], name='state')

        # network
        value_function_and_policy = ValueFunctionAndPolicy(np.squeeze(env.action_space.shape))
        value, dist = value_function_and_policy(state, training=training)

        # actor
        action_sample = dist.sample()
        action_sample_prob = dist.prob(action_sample)

        # training
        vars = tf.trainable_variables()
        updates = [tf.placeholder(var.dtype, var.shape) for var in vars]
        update_vars = tf.group(*[var.assign(update) for var, update in zip(vars, updates)])

        with tf.Session() as sess:
            us = ray.get(self.master.updates.remote())
            sess.run(update_vars, {update: u for update, u in zip(updates, us)})

            s = env.reset()
            t = 0
            total_reward = 0
            history = []

            for _ in itertools.count():
                a, a_prob = sess.run(
                    [action_sample, action_sample_prob], {state: np.expand_dims(s, 0), training: False})
                a, a_prob = map(lambda x: np.squeeze(x, 0), [a, a_prob])
                s_prime, r, d, _ = env.step(a)

                t += 1
                total_reward += r
                history.append((s, a, a_prob, r, d))

                if d:
                    ray.get(self.master.episode_metrics.remote(t, total_reward))

                    s = env.reset()
                    t = 0
                    total_reward = 0
                else:
                    s = s_prime

                if len(history) == self.config.horizon:
                    trajectory = build_trajectory(history, s)  # TODO: use s_prime?

                    us = ray.get(self.master.train.remote(trajectory))
                    sess.run(update_vars, {update: u for update, u in zip(updates, us)})

                    history = []


def build_trajectory(history, state_prime):
    state, action, action_prob, reward, done = [np.array(x) for x in zip(*history)]
    state_prime = np.array(state_prime)

    return state, action, action_prob, reward, state_prime, done


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--experiment-path', type=str, default='./tf_log/impala')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    ray.init()
    args = build_parser().parse_args()

    master = ray.remote(Master).remote(args)
    workers = [ray.remote(Worker).remote(master, args) for _ in range(os.cpu_count())]
    tasks = [w.train.remote() for w in workers]
    ray.get(tasks)


if __name__ == '__main__':
    main()

# TODO: test this
# def from_importance_weights(
#         log_rhos, discounts, rewards, values, bootstrap_value,
#         clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0,
#         name='vtrace_from_importance_weights'):
#     r"""V-trace from log importance weights.
#     Calculates V-trace actor critic targets as described in
#     "IMPALA: Scalable Distributed Deep-RL with
#     Importance Weighted Actor-Learner Architectures"
#     by Espeholt, Soyer, Munos et al.
#     In the notation used throughout documentation and comments, T refers to the
#     time dimension ranging from 0 to T-1. B refers to the batch size and
#     NUM_ACTIONS refers to the number of actions. This code also supports the
#     case where all tensors have the same number of additional dimensions, e.g.,
#     `rewards` is [T, B, C], `values` is [T, B, C], `bootstrap_value` is [B, C].
#     Args:
#       log_rhos: A float32 tensor of shape [T, B, NUM_ACTIONS] representing the log
#         importance sampling weights, i.e.
#         log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
#         on rhos in log-space for numerical stability.
#       discounts: A float32 tensor of shape [T, B] with discounts encountered when
#         following the behaviour policy.
#       rewards: A float32 tensor of shape [T, B] containing rewards generated by
#         following the behaviour policy.
#       values: A float32 tensor of shape [T, B] with the value function estimates
#         wrt. the target policy.
#       bootstrap_value: A float32 of shape [B] with the value function estimate at
#         time T.
#       clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
#         importance weights (rho) when calculating the baseline targets (vs).
#         rho^bar in the paper. If None, no clipping is applied.
#       clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
#         on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
#         None, no clipping is applied.
#       name: The name scope that all V-trace operations will be created in.
#     Returns:
#       A VTraceReturns namedtuple (vs, pg_advantages) where:
#         vs: A float32 tensor of shape [T, B]. Can be used as target to
#           train a baseline (V(x_t) - vs_t)^2.
#         pg_advantages: A float32 tensor of shape [T, B]. Can be used as the
#           advantage in the calculation of policy gradients.
#     """
#     log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
#     discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
#     rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
#     values = tf.convert_to_tensor(values, dtype=tf.float32)
#     bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)
#     if clip_rho_threshold is not None:
#         clip_rho_threshold = tf.convert_to_tensor(clip_rho_threshold,
#                                                   dtype=tf.float32)
#     if clip_pg_rho_threshold is not None:
#         clip_pg_rho_threshold = tf.convert_to_tensor(clip_pg_rho_threshold,
#                                                      dtype=tf.float32)
#
#     # Make sure tensor ranks are consistent.
#     rho_rank = log_rhos.shape.ndims  # Usually 2.
#     values.shape.assert_has_rank(rho_rank)
#     bootstrap_value.shape.assert_has_rank(rho_rank - 1)
#     discounts.shape.assert_has_rank(rho_rank)
#     rewards.shape.assert_has_rank(rho_rank)
#     if clip_rho_threshold is not None:
#         clip_rho_threshold.shape.assert_has_rank(0)
#     if clip_pg_rho_threshold is not None:
#         clip_pg_rho_threshold.shape.assert_has_rank(0)
#
#     with tf.name_scope(name, values=[
#         log_rhos, discounts, rewards, values, bootstrap_value]):
#         rhos = tf.exp(log_rhos)
#         if clip_rho_threshold is not None:
#             clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
#         else:
#             clipped_rhos = rhos
#
#         cs = tf.minimum(1.0, rhos, name='cs')
#         # Append bootstrapped value to get [v1, ..., v_t+1]
#         values_t_plus_1 = tf.concat(
#             [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
#         deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
#
#         # Note that all sequences are reversed, computation starts from the back.
#         sequences = (
#             tf.reverse(discounts, axis=[0]),
#             tf.reverse(cs, axis=[0]),
#             tf.reverse(deltas, axis=[0]),
#         )
#
#         # V-trace vs are calculated through a scan from the back to the beginning
#         # of the given trajectory.
#         def scanfunc(acc, sequence_item):
#             discount_t, c_t, delta_t = sequence_item
#             return delta_t + discount_t * c_t * acc
#
#         initial_values = tf.zeros_like(bootstrap_value)
#         vs_minus_v_xs = tf.scan(
#             fn=scanfunc,
#             elems=sequences,
#             initializer=initial_values,
#             parallel_iterations=1,
#             back_prop=False,
#             name='scan')
#         # Reverse the results back to original order.
#         vs_minus_v_xs = tf.reverse(vs_minus_v_xs, [0], name='vs_minus_v_xs')
#
#         # Add V(x_s) to get v_s.
#         vs = tf.add(vs_minus_v_xs, values, name='vs')
#
#         # Advantage for policy gradient.
#         vs_t_plus_1 = tf.concat([
#             vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
#         if clip_pg_rho_threshold is not None:
#             clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos, name='clipped_pg_rhos')
#         else:
#             clipped_pg_rhos = rhos
#         pg_advantages = (clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))
#
#         # Make sure no gradients backpropagated through the returned values.
#         return tf.stop_gradient(vs), tf.stop_gradient(pg_advantages)
