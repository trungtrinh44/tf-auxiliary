from tensorflow.contrib.rnn import LSTMBlockWrapper
from tensorflow.contrib.rnn.ops import gen_lstm_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import (array_ops, init_ops, math_ops, nn_ops,
                                   random_ops, rnn_cell_impl)
from tensorflow.python.platform import resource_loader


class WeighDropLSTMBlockFusedCell(LSTMBlockWrapper):
    """FusedRNNCell implementation of LSTM.

    This is an extremely efficient LSTM implementation, that uses a single TF op
    for the entire LSTM. It should be both faster and more memory-efficient than
    LSTMBlockCell defined above.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    The variable naming is consistent with `rnn_cell_impl.LSTMCell`.
    """

    def __init__(self,
                 num_units,
                 drop_w=0.0,
                 is_training=False,
                 forget_bias=1.0,
                 cell_clip=None,
                 use_peephole=False,
                 reuse=None,
                 name="lstm_fused_cell"):
        """Initialize the LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          cell_clip: clip the cell to this value. Default is no cell clipping.
          use_peephole: Whether to use peephole connections or not.
          reuse: (optional) boolean describing whether to reuse variables in an
            existing scope.  If not `True`, and the existing scope already has the
            given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.  By default this is "lstm_cell", for variable-name compatibility
            with `tf.nn.rnn_cell.LSTMCell`.
        """
        super(WeighDropLSTMBlockFusedCell, self).__init__(
            _reuse=reuse, name=name)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._cell_clip = cell_clip if cell_clip is not None else -1
        self._use_peephole = use_peephole
        self._is_training = is_training
        self._drop_w = drop_w

        # Inputs must be 3-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=3)

    @property
    def num_units(self):
        """Number of units in this cell (output dimension)."""
        return self._num_units

    def build(self, input_shape):
        input_size = input_shape[2].value
        self._W = self.add_variable(
            "W", [input_size, self._num_units*4]
        )
        self._U = self.add_variable(
            "U", [self._num_units, self._num_units*4]
        )
        self._bias = self.add_variable(
            "bias", [self._num_units * 4],
            initializer=init_ops.constant_initializer(0.0))
        if self._use_peephole:
            self._w_i_diag = self.add_variable("w_i_diag", [self._num_units])
            self._w_f_diag = self.add_variable("w_f_diag", [self._num_units])
            self._w_o_diag = self.add_variable("w_o_diag", [self._num_units])

        self.built = True

    def _call_cell(self,
                   inputs,
                   initial_cell_state=None,
                   initial_output=None,
                   dtype=None,
                   sequence_length=None):
        """Run this LSTM on inputs, starting from the given state.

        Args:
          inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
          initial_cell_state: initial value for cell state, shape `[batch_size,
            self._num_units]`
          initial_output: initial value of cell output, shape `[batch_size,
            self._num_units]`
          dtype: The data type for the initial state and expected output.
          sequence_length: Specifies the length of each sequence in inputs. An
            `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
            time_len)` or None.

        Returns:
          A pair containing:

          - Cell state (cs): A `3-D` tensor of shape `[time_len, batch_size,
                             output_size]`
          - Output (h): A `3-D` tensor of shape `[time_len, batch_size,
                        output_size]`
        """

        inputs_shape = inputs.get_shape().with_rank(3)
        time_len = inputs_shape[0].value
        if time_len is None:
            time_len = array_ops.shape(inputs)[0]

        if self._use_peephole:
            wci = self._w_i_diag
            wco = self._w_o_diag
            wcf = self._w_f_diag
        else:
            wci = wcf = wco = array_ops.zeros([self._num_units], dtype=dtype)

        if sequence_length is None:
            max_seq_len = math_ops.to_int64(time_len)
        else:
            max_seq_len = math_ops.to_int64(
                math_ops.reduce_max(sequence_length))
        if self._is_training:
            random_tensor = 1-self._drop_w
            random_tensor += random_ops.random_uniform(
                [self._num_units, 1], seed=None, dtype=dtypes.float32)
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            binary_tensor = math_ops.floor(random_tensor)
            self._U = math_ops.multiply(self._U, binary_tensor)
        self._kernel = array_ops.concat([self._W, self._U], axis=0)
        _, cs, _, _, _, _, h = gen_lstm_ops.block_lstm(
            seq_len_max=max_seq_len,
            x=inputs,
            cs_prev=initial_cell_state,
            h_prev=initial_output,
            w=self._kernel,
            wci=wci,
            wcf=wcf,
            wco=wco,
            b=self._bias,
            forget_bias=self._forget_bias,
            cell_clip=self._cell_clip,
            use_peephole=self._use_peephole)
        return cs, h
