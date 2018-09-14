import torch.nn as nn
import torch


# TODO: set trainable
# TODO:  initialization


class Network(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.dense_1 = nn.Linear(in_features, 32)
        self.dense_2 = nn.Linear(32, 32)

        nn.init.xavier_normal_(self.dense_1.weight)
        nn.init.xavier_normal_(self.dense_2.weight)

    def forward(self, input):
        input = self.dense_1(input)
        input = torch.tanh(input)

        input = self.dense_2(input)
        input = torch.tanh(input)

        return input


# class ValueFunction(tf.layers.Layer):
#     def __init__(self,
#                  trainable=True,
#                  name='value_function'):
#         super().__init__(name=name)
#
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
#             factor=2.0, mode='FAN_IN', uniform=False)
#         kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
#
#         self.net = Network(trainable=trainable)
#         self.dense = tf.layers.Dense(
#             1,
#             kernel_initializer=kernel_initializer,
#             kernel_regularizer=kernel_regularizer,
#             trainable=trainable)
#
#     def call(self, input, training):
#         input = self.net(input, training=training)
#         input = self.dense(input)
#         input = tf.squeeze(input, -1)
#
#         return input


class PolicyCategorical(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()

        self.net = Network(state_size)
        self.dense = nn.Linear(32, num_actions)

        nn.init.xavier_normal_(self.dense.weight)

    def forward(self, input):
        input = self.net(input)
        input = self.dense(input)

        dist = torch.distributions.Categorical(logits=input)

        return dist

# class PolicyNormal(tf.layers.Layer):
#     def __init__(self,
#                  num_actions,
#                  trainable=True,
#                  name='policy_categorical'):
#         super().__init__(name=name)
#
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
#             factor=2.0, mode='FAN_IN', uniform=False)
#         kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
#
#         self.net = Network(trainable=trainable)
#         self.dense = tf.layers.Dense(
#             num_actions * 2,
#             kernel_initializer=kernel_initializer,
#             kernel_regularizer=kernel_regularizer,
#             trainable=trainable)
#
#     def call(self, input, training):
#         input = self.net(input, training=training)
#         input = self.dense(input)
#
#         mu, sigma = tf.split(input, 2, -1)
#         sigma = tf.nn.softplus(sigma) + 1e-5
#         dist = tf.distributions.Normal(mu, sigma)
#
#         return dist
