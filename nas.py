import utils
import numpy as np
import os
import tensorflow as tf
import itertools
from mnist import MNIST


# TODO: opt settings
# TODO: bn, pool


def build_input_fns(dataset_path):
    def preprocess(images, labels):
        images = (np.array(images) / 255).astype(np.float32)
        labels = np.array(labels).astype(np.int32)
        # images = images.reshape((images.shape[0], 28, 28, 1))

        return images, labels

    def train_input_fn():
        ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        ds = ds.shuffle(1024)
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, 10)))
        ds = ds.batch(32)
        ds = ds.prefetch(None)

        return ds

    def eval_input_fn():
        ds = tf.data.Dataset.from_tensor_slices((eval_images, eval_labels))
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, 10)))
        ds = ds.batch(32)
        ds = ds.prefetch(None)

        return ds

    mnist = MNIST(dataset_path, gz=True)
    train_images, train_labels = mnist.load_training()
    eval_images, eval_labels = mnist.load_testing()
    train_images, train_labels = preprocess(train_images, train_labels)
    eval_images, eval_labels = preprocess(eval_images, eval_labels)

    return train_input_fn, eval_input_fn


def build_model_spec(actions, actions_per_layer):
    def build_layer(actions):
        return {
            'filters': [16, 32, 64, 128][actions[0]],
            # 'kernel_size': [3, 5, 7, 9][actions[1]],
            # 'strides': [1, 1, 2, 2][actions[2]],
            'activation': ['relu', 'tanh', 'selu', 'elu'][actions[1]],
            'dropout': [0.2, 0.4, 0.6, 0.8][actions[2]]
        }

    layers = [build_layer(actions[i: i + actions_per_layer]) for i in range(0, len(actions), actions_per_layer)]

    return layers


def build_estimator(model_spec, experiment_path):
    def model_spec_to_string(model_spec):
        s = ''
        for i, l in enumerate(model_spec):
            s += 'l{}(f={},a={},d={}),'.format(
                i + 1, l['filters'], l['activation'], l['dropout'])
        s = s[:-1]

        return s

    def model_fn(features, labels, mode, params):
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        input = features
        for i, l in enumerate(params['model_spec']):
            input = tf.layers.dense(
                input,
                l['filters'],
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)
            input = {
                'tanh': tf.nn.tanh,
                'relu': tf.nn.relu,
                'elu': tf.nn.elu,
                'selu': tf.nn.selu
            }[l['activation']](input)
            input = tf.layers.dropout(input, l['dropout'])

        input = tf.layers.dense(input, 10, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        logits = input

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss) + tf.losses.get_regularization_loss()

        global_step = tf.train.get_or_create_global_step()
        train_step = tf.train.AdamOptimizer().minimize(loss, global_step)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_step)

        elif mode == tf.estimator.ModeKeys.EVAL:
            metrics = {'accuracy': tf.metrics.mean(tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1)))}
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    model_dir = os.path.join(experiment_path, model_spec_to_string(model_spec))
    config = tf.estimator.RunConfig(model_dir=model_dir)
    params = {'model_spec': model_spec}

    estimator = tf.estimator.Estimator(model_fn, config=config, params=params)

    return estimator


def policy(num_actions, timesteps, name='policy'):
    with tf.name_scope(name):
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        cell = tf.nn.rnn_cell.LSTMCell(32, initializer=kernel_initializer)
        dense_out = tf.layers.Dense(
            num_actions,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

        input = tf.zeros((1, num_actions))
        state = cell.zero_state(1, input.dtype)

        actions = []
        logits = []

        for _ in range(timesteps):
            input, state = cell(input, state)
            logit = dense_out(input)
            logits.append(logit)
            dist = tf.distributions.Categorical(logit)
            action = dist.sample()
            actions.append(action)
            input = tf.one_hot(action, num_actions)

        actions = tf.stack(actions, 1)
        logits = tf.stack(logits, 1)
        dist = tf.distributions.Categorical(logits)

        return actions, dist


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, default='./tf_log/nas')
    parser.add_argument('--dataset-path', type=str, default=os.path.expanduser('~/Datasets/mnist'))
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--entropy-weight', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    utils.fix_seed(args.seed)
    experiment_path = args.experiment_path

    global_step = tf.train.get_or_create_global_step()

    # input
    b, t = 1, None
    actions = tf.placeholder(tf.int32, [b, t], name='actions')
    rewards = tf.placeholder(tf.float32, [b, t], name='rewards')

    # actor
    num_actions = 4
    layers = 4
    actions_per_layer = 3
    timesteps = layers * actions_per_layer
    action_samples, dist = policy(num_actions, timesteps)
    returns = utils.batch_return(rewards, gamma=args.gamma)
    advantages = tf.stop_gradient(returns)  # TODO: normalize advantages?
    actor_loss = -tf.reduce_mean(dist.log_prob(actions) * advantages)
    actor_loss -= args.entropy_weight * tf.reduce_mean(dist.entropy())

    # training
    loss = actor_loss + tf.losses.get_regularization_loss()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)

    # summary
    ep_reward = tf.placeholder(tf.float32, [])
    metrics, update_metrics = {}, {}
    metrics['loss'], update_metrics['loss'] = tf.metrics.mean(loss)
    metrics['ep_reward'], update_metrics['ep_reward'] = tf.metrics.mean(ep_reward)
    summary = tf.summary.merge([
        tf.summary.scalar('loss', metrics['loss']),
        tf.summary.scalar('ep_reward', metrics['ep_reward'])
    ])

    locals_init = tf.local_variables_initializer()

    hooks = [
        tf.train.CheckpointSaverHook(checkpoint_dir=experiment_path, save_steps=100)
    ]
    with tf.train.SingularMonitoredSession(checkpoint_dir=experiment_path, hooks=hooks) as sess, tf.summary.FileWriter(
            experiment_path) as writer:
        sess.run(locals_init)

        train_input_fn, eval_input_fn = build_input_fns(args.dataset_path)

        for _ in itertools.count():
            a = sess.run(action_samples)

            model_spec = build_model_spec(np.squeeze(a, 0), actions_per_layer)
            estimator = build_estimator(model_spec, experiment_path)
            ms = [estimator.evaluate(eval_input_fn)]
            for _ in range(5):
                estimator.train(train_input_fn)
                m = estimator.evaluate(eval_input_fn)
                ms.append(m)
            r = sum(m['accuracy'] for m in ms)

            _, _, step = sess.run(
                [train_step, update_metrics, global_step],
                {actions: a, rewards: [[0] * (timesteps - 1) + [r]], ep_reward: r})

            summ, metr = sess.run([summary, metrics])
            writer.add_summary(summ, step)
            writer.flush()
            sess.run(locals_init)


if __name__ == '__main__':
    main()
