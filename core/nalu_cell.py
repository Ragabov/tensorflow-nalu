import tensorflow as tf

from nalu.core.nac_cell import NacCell

class NaluCell(object):

    def __init__(self, input_shape, output_shape, scope = "nalu_cell"):
        """

        :param input_shape: input sample dimension
        :param output_shape: output dimension
        :param scope: the tensorflow scope used for variable's declaration
        """
        with tf.variable_scope(scope):
            self._add_sub_nac = NacCell(input_shape, output_shape, "add_subtract_NAC")
            self._mult_div_nac = NacCell(input_shape, output_shape, "multiply_divide_NAC")
            self._g = tf.get_variable(
                    "G",
                    [output_shape, input_shape],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=.01)
                )

    def __call__(self, input):
        """
        Performs forward propagation for the NAC cell

        :param input: a tensorflow input tensor
        :return: the outputs of the forward propagation
        """
        g = tf.sigmoid(tf.matmul(self._g, input))
        a = self._add_sub_nac(input)
        m = tf.asinh(self._mult_div_nac(tf.sinh((input))))
        y = tf.multiply(g, a) + tf.multiply(1 - g, m)

        return y