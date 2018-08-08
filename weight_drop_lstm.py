import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMStateTuple, RNNCell
from tensorflow.python.layers import base as base_layer
from tensorflow.python.platform import tf_logging as logging

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class WeightDropLSTMCell(RNNCell):
    """Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, drop=0.0,
                 state_is_tuple=True, activation=None, reuse=None, name=None):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          When restoring from CudnnLSTM-trained checkpoints, must use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(WeightDropLSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.tanh
        self._drop = drop

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._U = self.add_variable(
            _WEIGHTS_VARIABLE_NAME + '_U',
            shape=[h_depth, 4*self._num_units]
        )
        self._W = self.add_variable(
            _WEIGHTS_VARIABLE_NAME+'_W',
            shape=[input_depth, 4*self._num_units]
        )
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=tf.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size, self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size, 2 * self.state_size]`.
        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = tf.sigmoid
        one = tf.constant(1, dtype=tf.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=one)
        _U = tf.nn.dropout(self._U, 1-self._drop, name='U_drop')
        _kernel = tf.concat([self._W, _U], 0, name='concat_kernel')
        gate_inputs = tf.matmul(
            tf.concat([inputs, h], 1), _kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = tf.constant(
            self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = tf.add
        multiply = tf.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state


if __name__ == '__main__':
    import numpy as np
    from tensorflow.nn.rnn_cell import LSTMStateTuple
    np.random.seed(60)
    tf.set_random_seed(42)
    X = np.random.rand(6, 6, 3)
    inputs = tf.placeholder(shape=(None, None, 3),
                            dtype=tf.float32, name='token_ids')
    seq_len = tf.placeholder(
        shape=(None,), dtype=tf.int32, name='seq_len')
    reset_state = tf.placeholder(shape=(), dtype=tf.bool, name='reset_state')
    input_shape = tf.shape(inputs)
    inputs_ta = tf.TensorArray(
        size=input_shape[0], dtype=tf.float32, dynamic_size=True)
    inputs_ta = inputs_ta.unstack(inputs)
    cell = tf.nn.rnn_cell.MultiRNNCell(
        cells=[WeightDropLSTMCell(2, name='rnn_1'),
               WeightDropLSTMCell(2, name='rnn_2')]
    )
    zero_state = cell.zero_state(input_shape[1], tf.float32)
    all_states = tuple(
        LSTMStateTuple(
            c=tf.get_variable(shape=[1, 3], name='state_{}_c'.format(
                i), trainable=False),
            h=tf.get_variable(shape=[1, 3], name='state_{}_h'.format(
                i), trainable=False)) for i in range(len(zero_state))
    )

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
            next_cell_state = tf.cond(reset_state,
                                      lambda: zero_state,
                                      lambda: all_states)
        else:
            ops = [tf.assign(x.c, y.c, validate_shape=False) for x, y in zip(all_states, cell_state)] + [
                tf.assign(x.h, y.h, validate_shape=False) for x, y in zip(all_states, cell_state)]
            with tf.control_dependencies(ops):
                next_cell_state = tuple(
                    LSTMStateTuple(
                        c=tf.identity(x.c),
                        h=tf.identity(x.h)
                    ) for x in cell_state
                )
        elements_finished = (time >= seq_len)
        finished = tf.reduce_all(elements_finished)
        next_input = tf.cond(
            finished,
            lambda: tf.zeros(
                [input_shape[1], inputs.shape[-1]], dtype=tf.float32),
            lambda: inputs_ta.read(time))
        return (elements_finished, next_input, next_cell_state,
                emit_output, None)

    outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
    outputs = outputs_ta.stack()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for j in range(2):
        print('Epoch', j)
        for i in range(6):
            # if i > 0:
            #     print('Before state')
            #     print(sess.run(all_states))
            o = sess.run(final_state, feed_dict={
                inputs: X, seq_len: [6]*6, reset_state: i == 0})
            print('Final state:', o)
            # print('After state')
            print('Current state:', sess.run(all_states))
            n = [n.name for n in tf.get_default_graph().as_graph_def().node]
            print("No.of nodes: ", len(n), "\n")
    print(tf.trainable_variables())
