import tensorflow as tf
import resnet


class Q(tf.layers.Layer):
    def __init__(self, num_actions, name='q'):
        super().__init__(name=name)

        self._num_actions = num_actions
        self._kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        self._kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

    def build(self, input_shape):
        self._resnet = resnet.ResNet()
        self._avg_pool = tf.layers.AveragePooling2D((7, 5), 1)
        self._dense = tf.layers.Dense(
            self._num_actions, kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer)

        super().build(input_shape)

    def call(self, input, training):
        input = self._resnet(input, training=training)
        input = input['C5']
        input = self._avg_pool(input)
        input = tf.squeeze(input, (1, 2))
        input = self._dense(input)

        return input
