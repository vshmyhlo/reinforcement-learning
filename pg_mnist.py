import utils
import numpy as np
import os
import tensorflow as tf
from mnist import MNIST
import itertools
from tqdm import tqdm
from network import PolicyCategorical


def build_dataset(dataset_path):
    mnist = MNIST(dataset_path, gz=True)
    images, labels = mnist.load_training()
    images = (np.array(images) / 255).astype(np.float32)
    labels = np.array(labels).astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(32)
    ds = ds.prefetch(None)

    return ds


def build_parser():
    parser = utils.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--experiment-path', type=str, default='./tf_log/pg-mnist')
    parser.add_argument('--dataset-path', type=str, default=os.path.expanduser('~/Datasets/mnist'))
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--entropy-weight', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--monitor', action='store_true')

    return parser


def main():
    args = build_parser().parse_args()
    utils.fix_seed(args.seed)
    experiment_path = args.experiment_path

    global_step = tf.train.get_or_create_global_step()
    training = tf.placeholder(tf.bool, [], name='training')

    # input
    ds = build_dataset(args.dataset_path)
    images, labels = ds.make_one_shot_iterator().get_next()
    states = images

    # actor
    policy = PolicyCategorical(28 * 28)
    dist = policy(states, training=training)
    actions = tf.stop_gradient(dist.sample())
    rewards = tf.to_float(tf.equal(actions, labels))
    advantages = tf.stop_gradient(rewards)  # TODO: normalize advantages?
    actor_loss = -tf.reduce_mean(dist.log_prob(actions) * advantages)
    actor_loss -= args.entropy_weight * tf.reduce_mean(dist.entropy())

    # training
    loss = actor_loss + tf.losses.get_regularization_loss()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)

    # summary
    metrics, update_metrics = {}, {}
    metrics['loss'], update_metrics['loss'] = tf.metrics.mean(loss)
    metrics['ep_reward'], update_metrics['ep_reward'] = tf.metrics.mean(rewards)
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

        for _ in tqdm(itertools.count()):
            _, _, step = sess.run([train_step, update_metrics, global_step])

            if step % 100 == 0:
                summ, metr = sess.run([summary, metrics])
                writer.add_summary(summ, step)
                writer.flush()
                sess.run(locals_init)


if __name__ == '__main__':
    main()
