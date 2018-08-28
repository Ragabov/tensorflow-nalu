import tensorflow as tf
from core.nac_cell import NacCell
from core.nalu_cell import NaluCell


class NaluLayer(object):

    def __init__(self, input_shape, output_shape, hidden_shape = 100, n_cells = 1,
                 core_cell_type="nalu", scope="nalu_layer"):
        """

        :param input_shape: input sample dimension
        :param output_shape: output sample dimension
        :param hidden_shape: the dimension of the hidden core cells if any
        :param n_cells: the number of cells to construct, if n_cells > 1, then all the inner cells will have an output
        shape of hidden_shape
        :param core_cell_type: the core cell type to use, this can be either "nalu" or "nac"
        :param scope: the tensorflow scope used for variable's declaration
        """
        with tf.variable_scope(scope):
            core_cell = None
            if core_cell_type == "nalu":
                core_cell = NaluCell
            elif core_cell_type == "nac":
                core_cell = NacCell
            else:
                raise ValueError("core cell types can be either 'nalu' or 'nac'")

            self.cells = []
            for i in range(n_cells):
                self.cells.append(
                    core_cell(
                        input_shape if i == 0 else hidden_shape,
                        output_shape if i == n_cells - 1 else hidden_shape,
                        core_cell_type + str(i)
                    )
                )

    def __call__(self, input):
        """
        Performs forward propagation

        :param input: a tensorflow input tensor of shape [BATCH_SIZE X input_shape]
        :return: the outputs of the forward propagation of shape [BATCH_SIZE X output_shape]
        """
        output = None
        input = tf.transpose(input)
        for i, cell in enumerate(self.cells):
            if i == 0:
                output = cell(input)
            else :
                output = cell(output)

        return tf.transpose(output)