import tensorflow as tf


class Network(tf.layers.Layer):
    def __init__(self,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 trainable=True,
                 name='network'):
        super().__init__(name=name)

        self.dense_1 = tf.layers.Dense(
            32,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            trainable=trainable)

        self.dense_2 = tf.layers.Dense(
            32,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            trainable=trainable)

    def call(self, input, training):
        input = self.dense_1(input)
        input = tf.nn.elu(input)

        input = self.dense_2(input)
        input = tf.nn.elu(input)

        return input


class ValueFunction(tf.layers.Layer):
    def __init__(self,
                 trainable=True,
                 name='value_function'):
        super().__init__(name=name)

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.net = Network(trainable=trainable)
        self.dense = tf.layers.Dense(
            1,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            trainable=trainable)

    def call(self, input, training):
        input = self.net(input, training=training)
        input = self.dense(input)
        input = tf.squeeze(input, -1)

        return input


class PolicyCategorical(tf.layers.Layer):
    def __init__(self,
                 num_actions,
                 trainable=True,
                 name='policy_categorical'):
        super().__init__(name=name)

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.net = Network(trainable=trainable)
        self.dense = tf.layers.Dense(
            num_actions,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            trainable=trainable)

    def call(self, input, training):
        input = self.net(input, training=training)
        input = self.dense(input)

        dist = tf.distributions.Categorical(logits=input)

        return dist


class PolicyNormal(tf.layers.Layer):
    def __init__(self,
                 num_actions,
                 trainable=True,
                 name='policy_categorical'):
        super().__init__(name=name)

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.net = Network(trainable=trainable)
        self.dense = tf.layers.Dense(
            num_actions * 2,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            trainable=trainable)

    def call(self, input, training):
        input = self.net(input, training=training)
        input = self.dense(input)

        mu, sigma = tf.split(input, 2, -1)
        sigma = tf.nn.softplus(sigma) + 1e-5  # TODO: softplus?

        dist = tf.distributions.Normal(mu, sigma)

        return dist
