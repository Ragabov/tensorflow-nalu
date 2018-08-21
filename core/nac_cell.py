from random import uniform

import tensorflow as tf


class NacCell(object):

    def __init__(self, input_shape, output_shape, scope = "nac_cell"):
        """

        :param input_shape: input sample dimension
        :param output_shape: output dimension
        :param scope: the tensorflow scope used for variable's declaration
        """
        with tf.variable_scope(scope):
            self._Wt = tf.get_variable(
                "Wt",
                [output_shape, input_shape],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=.01)
            )
            self._M = tf.get_variable(
                "M",
                [output_shape, input_shape],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=.01)
            )
            self.W = tf.multiply(tf.tanh(self._Wt), tf.sigmoid(self._Wt))

    def __call__(self, input):
        """
        Performs forward propagation for the NAC cell

        :param input: a tensorflow input tensor
        :return: the outputs of the forward propagation
        """
        return tf.matmul(self.W, input)
