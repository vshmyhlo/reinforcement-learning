import torch
import torch.nn as nn

SIZE = 32


class Model(nn.Module):
    def __init__(self, policy=None):
        super().__init__()
       
        self.policy = policy


class Network(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.l_1 = nn.Linear(in_features, SIZE)
        self.l_2 = nn.Linear(SIZE, SIZE)
        self.act = nn.ReLU(inplace=True)

        nn.init.xavier_normal_(self.l_1.weight)
        nn.init.xavier_normal_(self.l_2.weight)

    def forward(self, input):
        input = self.l_1(input)
        input = self.act(input)
        input = self.l_2(input)
        input = self.act(input)

        return input


class ValueFunction(nn.Module):
    def __init__(self, state_size):
        super().__init__()

        self.net = Network(state_size)
        self.linear = nn.Linear(SIZE, 1)

        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, input):
        input = self.net(input)
        input = self.linear(input)
        input = input.squeeze(-1)

        return input


class PolicyCategorical(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()

        self.net = Network(state_size)
        self.dense = nn.Linear(SIZE, num_actions)

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
