import tensorflow as tf
from embed_dropout import embedding_dropout
from tensorflow.nn.rnn_cell import LSTMStateTuple
from weight_drop_lstm import WeighDropLSTMBlockFusedCell


class LanguageModel():
    def __init__(self, vocab_size,
                 rnn_layers,
                 drop_e,
                 is_training,
                 parallel_iterations=16,
                 custom_getter=None, reuse=False, name='LanguageModel'):
        self.vocab_size = vocab_size
        self.rnn_layers = rnn_layers
        self.drop_e = drop_e
        self.name = name
        self.is_training = is_training
        self.custom_getter = custom_getter
        self.reuse = reuse
        self.parallel_iterations = parallel_iterations

    def build_model(self):
        with tf.variable_scope(self.name, custom_getter=self.custom_getter, reuse=self.reuse):
            # Inputs must be sequences of token ids with shape [time, batch, depth]
            # rnn_layers is a list of dictionaries, each contains all the parameters of the __get_rnn_cell function.
            self.inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name='token_ids')
            self.seq_lens = tf.placeholder(dtype=tf.int32,
                                           shape=[None],
                                           name='seq_lens')
            self.seq_masks = tf.transpose(tf.sequence_mask(self.seq_lens,
                                                           dtype=tf.float32),
                                          [1, 0])
            self.reset_state = tf.placeholder(dtype=tf.bool,
                                              shape=[],
                                              name='reset_state')
            with tf.device('/cpu:0'):
                self._W = tf.get_variable(
                    shape=[self.vocab_size, self.rnn_layers[0]['input_size']],
                    initializer=tf.glorot_uniform_initializer(),
                    name="embedding_weight")
                if self.is_training and self.drop_e < 1:
                    self._W = embedding_dropout(self._W, dropout=self.drop_e)
                self._embedding = tf.nn.embedding_lookup(
                    self._W, self.inputs
                )
            input_shape = tf.shape(self.inputs)
            ops = []
            inputs = self._embedding
            for idx, l in enumerate(self.rnn_layers):
                cell = WeighDropLSTMBlockFusedCell(
                    num_units=l['units'],
                    is_training=self.is_training,
                    drop_w=l.get('drop_w', 0.0)
                )
                saved_state = LSTMStateTuple(c=tf.get_variable(shape=[1, l['units']], name='c_'+str(idx), trainable=False),
                                             h=tf.get_variable(
                                                 shape=[1, l['units']], name='h_'+str(idx), trainable=False))
                zeros = tf.zeros(
                    [input_shape[1], l['units']], dtype=tf.float32)
                zero_state = LSTMStateTuple(c=zeros, h=zeros)

                def if_true():
                    return zero_state

                def if_false():
                    return saved_state
                drop_i = l.get('drop_i', 0.0)
                if self.is_training and drop_i > 0.0:
                    inputs = tf.nn.dropout(
                        x=inputs,
                        keep_prob=1-drop_i,
                        noise_shape=[1, input_shape[1], inputs.shape[-1]],
                        name='drop_i_'+str(idx)
                    )
                outputs, state = cell(
                    inputs=inputs,
                    initial_state=tf.cond(self.reset_state, if_true, if_false),
                    sequence_length=self.seq_lens
                )
                drop_o = l.get('drop_o', 0.0)
                if self.is_training and drop_o > 0.0:
                    outputs = tf.nn.dropout(
                        x=outputs,
                        keep_prob=1-drop_o,
                        noise_shape=[1, input_shape[1], outputs.shape[-1]],
                        name='drop_o_'+str(idx)
                    )
                ops.append(tf.assign(saved_state.c,
                                     state.c, validate_shape=False))
                ops.append(tf.assign(saved_state.h,
                                     state.h, validate_shape=False))
                inputs = outputs
            with tf.control_dependencies(ops):
                self.rnn_outputs = tf.identity(inputs, name='rnn_outputs')
            self.decoder = tf.nn.xw_plus_b(
                tf.reshape(self.rnn_outputs,
                           [input_shape[0]*input_shape[1], self.rnn_layers[0]['input_size']]),
                tf.transpose(self._W, [1, 0]),
                tf.get_variable(name='decoder_b',
                                shape=[self.vocab_size],
                                initializer=tf.glorot_uniform_initializer())
            )
            self.decoder = tf.reshape(
                self.decoder, [input_shape[0], input_shape[1], self.vocab_size])


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)
    tf.set_random_seed(42)
    V = 50
    model = LanguageModel(
        vocab_size=V,
        rnn_layers=[
            {'units': 3, 'input_size': 4, 'drop_i': 0.0, 'drop_w': 0.0},
            {'units': 2, 'input_size': 3, 'drop_w': 0.0},
            {'units': 4, 'input_size': 2, 'drop_o': 0.0, 'drop_w': 0.0}
        ],
        is_training=True,
        drop_e=0.0
    )
    model.build_model()
    words = np.random.random_integers(low=0, high=V-1, size=(10, 5))
    print(words)
    sess = tf.Session()
    ema = tf.train.ExponentialMovingAverage(0.998)
    var_class = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.name)
    ema_op = ema.apply(var_class)
    for v in var_class:
        print(v.op.name)
        print(ema.average(v))
    sess.run(tf.global_variables_initializer())
    for j in range(2):
        print('Epoch', j)
        for i in range(6):
            o = sess.run(model.decoder,
                         feed_dict={
                             model.inputs: words,
                             model.seq_lens: [10, 8, 7, 9, 6],
                             model.reset_state: i == 0
                         })
            print('Outputs', j, ':', o)
            n = [n.name for n in tf.get_default_graph().as_graph_def().node]
            print("No.of nodes: ", len(n), "\n")
