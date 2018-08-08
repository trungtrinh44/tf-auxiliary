import tensorflow as tf
from weight_drop_lstm import WeightDropLSTMCell
from embed_dropout import embedding_dropout
from tensorflow.nn.rnn_cell import LSTMStateTuple


class LanguageModel():
    def __get_rnn_cell(self, units, input_size, drop_i=0.0, drop_w=0.0, drop_o=0.0):
        return tf.nn.rnn_cell.DropoutWrapper(
            cell=WeightDropLSTMCell(units, drop=drop_w, state_is_tuple=True),
            input_keep_prob=1-drop_i,
            output_keep_prob=1-drop_o,
            state_keep_prob=1.0,
            variational_recurrent=True,
            dtype=tf.float32,
            input_size=input_size
        )

    def __init__(self, vocab_size, rnn_layers, drop_e, name='LanguageModel'):
        self.vocab_size = vocab_size
        self.rnn_layers = rnn_layers
        self.drop_e = drop_e
        self.name = name

    def build_model(self):
        with tf.variable_scope(self.name):
            # Inputs must be sequences of token ids with shape [time, batch, depth]
            # rnn_layers is a list of dictionaries, each contains all the parameters of the __get_rnn_cell function.
            self.inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name='token_ids')
            self.seq_lens = tf.placeholder(dtype=tf.int32,
                                           shape=[None],
                                           name='seq_lens')
            self.reset_state = tf.placeholder(dtype=tf.bool,
                                              shape=[],
                                              name='reset_state')
            self._W = tf.Variable(
                tf.random_uniform(
                    [self.vocab_size, self.rnn_layers[0]['input_size']],
                    -1.0, 1.0),
                name="embedding_weight")
            self._embedding = tf.nn.embedding_lookup(
                embedding_dropout(self._W, dropout=self.drop_e), self.inputs
            )
            self._cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.__get_rnn_cell(**l) for l in self.rnn_layers],
                state_is_tuple=True
            )
            input_shape = tf.shape(self.inputs)
            inputs_ta = tf.TensorArray(
                size=input_shape[0],
                dtype=tf.float32,
                dynamic_size=True
            )
            inputs_ta = inputs_ta.unstack(self._embedding)
            self._zero_state = self._cell.zero_state(
                input_shape[1],
                tf.float32)
            self._all_states = tuple(
                LSTMStateTuple(
                    c=tf.get_variable(shape=[1, 3], name='state_{}_c'.format(
                        i), trainable=False),
                    h=tf.get_variable(shape=[1, 3], name='state_{}_h'.format(
                        i), trainable=False)) for i in range(len(self._zero_state))
            )

            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output  # == None for time == 0
                if cell_output is None:  # time == 0
                    next_cell_state = tf.cond(self.reset_state,
                                              lambda: self._zero_state,
                                              lambda: self._all_states)
                else:
                    next_cell_state = tuple(
                        LSTMStateTuple(
                            c=tf.assign(x.c, y.c, validate_shape=False),
                            h=tf.assign(x.h, y.h, validate_shape=False)
                        )
                        for x, y in zip(self._all_states, cell_state)
                    )
                elements_finished = (time >= self.seq_lens)
                finished = tf.reduce_all(elements_finished)
                next_input = tf.cond(
                    finished,
                    lambda: tf.zeros(
                        [input_shape[1], self._W.shape[-1]], dtype=tf.float32),
                    lambda: inputs_ta.read(time))
                return (elements_finished, next_input, next_cell_state,
                        emit_output, None)
            self._loop_fn = loop_fn
            outputs_ta, final_state, _ = tf.nn.raw_rnn(
                self._cell, self._loop_fn)
            self.outputs = outputs_ta.stack()
            self.final_state = final_state


if __name__ == '__main__':
    import numpy as np
    V = 50
    model = LanguageModel(
        vocab_size=V,
        rnn_layers=[
            {'units': 3, 'input_size': 4, 'drop_i': 0.0, 'drop_w': 0.0},
            {'units': 2, 'input_size': 3, 'drop_w': 0.0},
            {'units': 2, 'input_size': 2, 'drop_o': 0.0, 'drop_w': 0.0}
        ],
        drop_e=0.0
    )
    model.build_model()
    np.random.seed(42)
    tf.set_random_seed(42)
    words = np.random.random_integers(low=0, high=V-1, size=(10, 5))
    print(words)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for j in range(2):
        print('Epoch', j)
        for i in range(6):
            o = sess.run(model.outputs,
                         feed_dict={
                             model.inputs: words,
                             model.seq_lens: [10]*5,
                             model.reset_state: i == 0
                         })
            print('Outputs', o)
            n = [n.name for n in tf.get_default_graph().as_graph_def().node]
            print("No.of nodes: ", len(n), "\n")
    print(tf.trainable_variables())