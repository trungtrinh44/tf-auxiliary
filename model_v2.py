# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleLSTMCell, CudnnLSTM

from embed_dropout import embedding_dropout
from layer_wise_lr import apply_custom_lr

LSTM_SAVED_STATE = 'LSTM_SAVED_STATE'


def __last_relevant(inputs, seq_lens, name):
    # Input shape: [seq_lens, batch_size, dims]
    # Select last output: batch_size * (seq_lens - 1) + batch_index
    with tf.variable_scope(name):
        s = tf.shape(inputs)
        max_lens, batch_size, input_dims = s[0], s[1], s[2]
        indices = batch_size * (seq_lens-1) + tf.range(start=0, limit=batch_size, delta=1, dtype=tf.int32)
        flat = tf.reshape(inputs, [max_lens*batch_size, input_dims])
        relevant = tf.gather(flat, indices)
        relevant = tf.reshape(relevant, [batch_size, input_dims])
        return relevant


def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
    carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
    transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
    return carry_gate * transform_gate + (1.0 - carry_gate) * x


class LanguageModel():
    def __init__(self, char_vocab_size, char_vec_size,
                 char_cnn_options,
                 vocab_size,
                 rnn_layers,
                 drop_e,
                 is_training,
                 projection_dims,
                 skip_connection,
                 fine_tune_lr=None, is_encoding=False,
                 custom_getter=None, reuse=False, name='LanguageModel'):
        self.vocab_size = vocab_size
        self.char_vocab_size = char_vocab_size
        self.rnn_layers = rnn_layers
        self.drop_e = drop_e
        self.name = name
        self.is_training = is_training
        self.custom_getter = custom_getter
        self.reuse = reuse
        self.fine_tune_lr = fine_tune_lr
        self.char_cnn_options = char_cnn_options
        self.char_vec_size = char_vec_size
        self.is_encoding = is_encoding
        self.projection_dims = projection_dims
        self.skip_connection = skip_connection

    def __build_uni_model(self, inputs, name, reuse=None):
        model = {}
        with tf.variable_scope(name, reuse=reuse if reuse else self.reuse):
            s = tf.shape(inputs)
            T, B = s[0], s[1]
            input_shape = (T, B, inputs.shape[-1])
            ops = []
            layer_outputs = [inputs]
            for idx, l in enumerate(self.rnn_layers):
                cell = CudnnLSTM(
                    num_layers=1,
                    num_units=l['units'],
                    input_mode='linear_input',
                    direction='unidirectional',
                    dropout=0.0
                )
                saved_state = (tf.get_variable(shape=[1, 1, l['units']], name='c_'+str(idx), trainable=False),
                               tf.get_variable(shape=[1, 1, l['units']], name='h_'+str(idx), trainable=False))
                for x in saved_state:
                    tf.add_to_collection(LSTM_SAVED_STATE, x)
                zeros = tf.zeros(
                    [1, input_shape[1], l['units']], dtype=tf.float32)
                zero_state = (zeros, zeros)

                def if_true():
                    return zero_state

                def if_false():
                    return saved_state
                drop_i = l.get('drop_i', 0.0)
                if self.is_training and drop_i > 0.0:
                    inputs = tf.nn.dropout(
                        x=inputs,
                        keep_prob=1-drop_i,
                        noise_shape=[
                            1, input_shape[1], inputs.shape[-1]],
                        name='drop_i_'+str(idx)
                    )
                cell.build(inputs.shape)
                wdrop = l.get('wdrop', 0.0)
                if self.is_training and wdrop > 0.0:
                    cell_var = cell.variables[0]
                    h_var_backup = tf.Variable(
                        initial_value=tf.zeros(
                            shape=[4*l['units'], l['units']]),
                        trainable=False,
                        name='h_var_backup_'+str(idx)
                    )
                    h_var = cell_var[inputs.shape[-1]
                                     * l['units']*4:-l['units']*8]
                    h_var = tf.reshape(
                        h_var, [4*l['units'], l['units']]) + h_var_backup
                    keep_prob = 1-wdrop
                    random_tensor = keep_prob
                    random_tensor += tf.random_uniform(
                        [4*l['units'], 1], dtype=h_var.dtype)
                    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
                    binary_tensor = tf.floor(random_tensor)
                    new_h_var = tf.multiply(h_var, binary_tensor)
                    new_h_var = tf.reshape(
                        new_h_var, [4*l['units']*l['units']])
                    h_var_backup = tf.assign(
                        h_var_backup,
                        tf.multiply(h_var, tf.subtract(
                            1.0, binary_tensor)),
                        validate_shape=True,
                        use_locking=True,
                        name='assign_h_var_backup_' + str(idx)
                    )
                    new_cell_var = tf.concat([
                        cell_var[:inputs.shape[-1]*l['units']*4],
                        new_h_var,
                        cell_var[-l['units']*8:]
                    ], axis=0, name='new_cell_var_' + str(idx))
                    op = tf.assign(
                        cell_var,
                        new_cell_var,
                        validate_shape=True,
                        use_locking=True,
                        name='assign_new_cell_var_' + str(idx)
                    )
                    with tf.control_dependencies([op, h_var_backup]):
                        outputs, state = cell.call(
                            inputs=inputs,
                            initial_state=tf.cond(
                                self.reset_state, if_true, if_false),
                            training=self.is_training
                        )
                else:
                    outputs, state = cell.call(
                        inputs=inputs,
                        initial_state=tf.cond(
                            self.reset_state, if_true, if_false),
                        training=self.is_training
                    )
                drop_o = l.get('drop_o', 0.0)
                if self.is_training and drop_o > 0.0:
                    outputs = tf.nn.dropout(
                        x=outputs,
                        keep_prob=1-drop_o,
                        noise_shape=[1, input_shape[1],
                                     outputs.shape[-1]],
                        name='drop_o_'+str(idx)
                    )
                if isinstance(self.projection_dims, int) and self.projection_dims > 0:
                    outputs = tf.reshape(outputs, (T * B, outputs.shape[-1]))
                    w_proj = tf.get_variable(name='w_proj_{}'.format(idx), shape=(outputs.shape[-1], self.projection_dims), initializer=tf.glorot_uniform_initializer())
                    b_proj = tf.get_variable(name='b_proj_{}'.format(idx), shape=(self.projection_dims,), initializer=tf.zeros_initializer())
                    outputs = tf.matmul(outputs, w_proj) + b_proj
                    outputs = tf.reshape(outputs, (T, B, self.projection_dims))
                    if idx > 0 and self.skip_connection:
                        outputs = tf.add(outputs, inputs, name='skip_{}'.format(idx))
                for x in saved_state:
                    x.validate_shape = False
                ops.append(tf.assign(saved_state[0],
                                     state[0], validate_shape=False))
                ops.append(tf.assign(saved_state[1],
                                     state[1], validate_shape=False))
                inputs = outputs
                if isinstance(self.fine_tune_lr, list):
                    outputs = apply_custom_lr(self.fine_tune_lr[idx])(outputs)
                layer_outputs.append(outputs)
            model['layer_outputs'] = layer_outputs
            ops = tf.group(ops)
            with tf.control_dependencies([ops]):
                rnn_outputs = tf.multiply(
                    inputs,
                    tf.expand_dims(self.seq_masks, axis=-1),
                    name='rnn_outputs'
                )
            model['rnn_outputs'] = rnn_outputs
            if not self.is_encoding:
                decoder = tf.nn.xw_plus_b(
                    tf.reshape(rnn_outputs,
                               (input_shape[0] * input_shape[1], rnn_outputs.shape[-1])),
                    self.share_decode_W,
                    self.share_decode_b
                )
                decoder = tf.reshape(
                    decoder, (input_shape[0], input_shape[1], self.vocab_size))
                model['decoder'] = decoder
            return model

    def __build_word_embedding(self, inputs, reuse, name='word_embedding'):
        with tf.variable_scope(name, reuse=reuse):
            # Reshape from [T, B, C] to [T * B, C]
            s = tf.shape(inputs)
            T, B = s[0], s[1]
            inputs = tf.reshape(inputs, (T * B, -1))
            with tf.device('/cpu:0'):
                W = tf.get_variable(
                    shape=[self.char_vocab_size, self.char_vec_size],
                    initializer=tf.glorot_uniform_initializer(),
                    name="embedding_weight")
                if self.is_training and self.drop_e > 0.0:
                    W = embedding_dropout(W, dropout=self.drop_e)
                char_embed = tf.nn.embedding_lookup(
                    W, inputs
                )
            conv_out = []
            for fsz, num in self.char_cnn_options['layers']:
                x = tf.layers.conv1d(
                    char_embed,
                    num,
                    fsz,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    padding='same'
                )
                x = tf.reduce_max(x, axis=1)
                conv_out.append(x)
            embedding = tf.concat(
                conv_out,
                axis=-1
            )
            nfilters = sum(x for _, x in self.char_cnn_options['layers'])
            for i in range(self.char_cnn_options.get('n_highways', 0)):
                ww_carry = tf.get_variable(name='ww_carry_{}'.format(i), shape=(nfilters, nfilters), initializer=tf.glorot_uniform_initializer())
                bb_carry = tf.get_variable(name='bb_carry_{}'.format(i), shape=(nfilters,), initializer=tf.zeros_initializer())
                ww_tr = tf.get_variable(name='ww_tr_{}'.format(i), shape=(nfilters, nfilters), initializer=tf.glorot_uniform_initializer())
                bb_tr = tf.get_variable(name='bb_tr_{}'.format(i), shape=(nfilters,), initializer=tf.zeros_initializer())
                embedding = high(embedding, ww_carry, bb_carry, ww_tr, bb_tr)
            if isinstance(self.projection_dims, int) and self.projection_dims != nfilters:
                w_proj = tf.get_variable(name='w_proj', shape=(nfilters, self.projection_dims), initializer=tf.glorot_uniform_initializer())
                b_proj = tf.get_variable(name='b_proj', shape=(self.projection_dims,), initializer=tf.zeros_initializer())
                embedding = tf.matmul(embedding, w_proj) + b_proj
                embedding = tf.reshape(
                    embedding,
                    (T, B, self.projection_dims)
                )
            else:
                embedding = tf.reshape(
                    embedding,
                    (T, B, nfilters)
                )
            return embedding

    def __build_language_model(self):
        self.fw_inputs = tf.placeholder(dtype=tf.int32,
                                        shape=[None, None, None],
                                        name='fw_inputs')
        self.bw_inputs = tf.placeholder(dtype=tf.int32,
                                        shape=[None, None, None],
                                        name='bw_inputs')
        self.share_decode_W = tf.get_variable(
            name='decode_W',
            shape=(self.projection_dims if isinstance(self.projection_dims, int) and self.projection_dims > 0 else self.rnn_layers[-1]['units'], self.vocab_size),
            initializer=tf.glorot_uniform_initializer()
        )
        self.share_decode_b = tf.get_variable(
            name='decode_b',
            shape=(self.vocab_size,),
            initializer=tf.zeros_initializer()
        )
        self.fw_model = self.__build_uni_model(
            self.__build_word_embedding(self.fw_inputs, reuse=self.reuse),
            'LMFW')
        self.bw_model = self.__build_uni_model(
            self.__build_word_embedding(self.bw_inputs, reuse=True),
            'LMBW')

    def __build_encoding_model(self):
        self.fw_inputs = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None, None),
                                        name='fw_inputs')
        self.bw_inputs = tf.reverse_sequence(
            input=self.fw_inputs,
            seq_lengths=self.seq_lens,
            seq_axis=0,
            batch_axis=1,
            name='bw_inputs'
        )
        self.inputs = self.fw_inputs
        self.fw_model = self.__build_uni_model(self.__build_word_embedding(self.fw_inputs, reuse=self.reuse), 'LMFW')
        self.bw_model = self.__build_uni_model(self.__build_word_embedding(self.bw_inputs, reuse=True), 'LMBW')
        batch_size = tf.shape(self.fw_inputs)[1]
        max_len = tf.reduce_max(self.seq_lens)
        start_i = tf.constant(0)
        start_max_val = [tf.ones(shape=(batch_size, fw.shape[-1]+bw.shape[-1]))*-1e6 for fw, bw in zip(self.fw_model['layer_outputs'], self.bw_model['layer_outputs'])]
        start_mean_val = [tf.zeros(shape=(batch_size, fw.shape[-1]+bw.shape[-1])) for fw, bw in zip(self.fw_model['layer_outputs'], self.bw_model['layer_outputs'])]
        self.bptt = tf.placeholder(dtype=tf.int32, shape=(), name='bptt')

        def cond(i, max_vals, mean_vals, fw_inputs, bw_inputs):
            return i < max_len

        def body(i, max_vals, mean_vals, fw_inputs, bw_inputs):
            i_to = tf.minimum(i+self.bptt, max_len)
            sliced_fw_inputs = fw_inputs[i:i_to]
            sliced_bw_inputs = bw_inputs[i:i_to]
            self.fw_model = self.__build_uni_model(self.__build_word_embedding(sliced_fw_inputs, reuse=True), 'LMFW', True)
            self.bw_model = self.__build_uni_model(self.__build_word_embedding(sliced_bw_inputs, reuse=True), 'LMBW', True)
            mask = tf.expand_dims(tf.transpose(tf.sequence_mask(tf.minimum(self.seq_lens-i, self.bptt), dtype=tf.float32), (1, 0)), axis=-1)
            new_max_vals = []
            new_mean_vals = []
            for max_val, mean_val, fw, bw in zip(max_vals, mean_vals, self.fw_model['layer_outputs'], self.bw_model['layer_outputs']):
                bw = tf.reverse_sequence(
                    input=bw,
                    seq_lengths=self.seq_lens,
                    seq_axis=0,
                    batch_axis=1
                )
                outputs = tf.concat((fw, bw), axis=-1)
                max_outputs = outputs * mask + (1 - mask) * -1e6
                max_val = tf.maximum(max_val, tf.reduce_max(max_outputs, axis=0))
                mean_outputs = outputs * mask
                mean_val = (mean_val * tf.expand_dims(tf.to_float(tf.minimum(i, self.seq_lens)), axis=-1) +
                            tf.reduce_sum(mean_outputs, axis=0)) / tf.expand_dims(tf.to_float(tf.minimum(i_to, self.seq_lens)), axis=-1)
                new_max_vals.append(max_val)
                new_mean_vals.append(mean_val)
            return i_to, new_max_vals, new_mean_vals, fw_inputs, bw_inputs
        _, self.loop_layerwise_max, self.loop_layerwise_avg, _, _ = tf.while_loop(cond, body, [start_i, start_max_val, start_mean_val, self.fw_inputs, self.bw_inputs])
        self.timewise_outputs = []
        self.layerwise_avg = []
        self.layerwise_max = []
        indices = tf.range(start=0, limit=tf.shape(self.seq_lens)[0], delta=1, dtype=tf.int32)
        indices = tf.stack((self.seq_lens - 1, indices), axis=-1)
        self.encode_outputs = []
        for idx, (fw, bw) in enumerate(zip(self.fw_model['layer_outputs'], self.bw_model['layer_outputs'])):
            bw = tf.reverse_sequence(
                input=bw,
                seq_lengths=self.seq_lens,
                seq_axis=0,
                batch_axis=1,
                name='bw_outputs_{}'.format(idx)
            )
            fwo = tf.gather_nd(params=fw, indices=indices)
            bwo = tf.gather_nd(params=bw, indices=indices)
            to = tf.multiply(
                tf.concat((fw, bw), axis=-1),
                tf.expand_dims(self.seq_masks, axis=-1)
            )
            self.timewise_outputs.append(to)
            self.layerwise_avg.append(tf.truediv(tf.reduce_sum(to, axis=0, keepdims=False), tf.to_float(self.seq_lens)))
            self.layerwise_max.append(tf.reduce_max(to, axis=0, keepdims=False))
            self.encode_outputs.append(tf.concat((fwo, bwo), axis=-1))
        self.concated_encode_output = tf.concat(self.encode_outputs, -1, name='concated_encode_output')
        self.concated_avg_output = tf.concat(self.layerwise_avg, -1, name='concated_avg_output')
        self.concated_max_output = tf.concat(self.layerwise_max, -1, name='concated_max_output')
        self.concated_timewise_output = tf.stack(self.timewise_outputs, axis=-1, name='concated_timewise_output')

    def build_model(self):
        with tf.variable_scope(self.name, custom_getter=self.custom_getter, reuse=self.reuse):
            # Inputs must be sequences of token ids with shape [time, batch, depth]
            # rnn_layers is a list of dictionaries, each contains all the parameters of the __get_rnn_cell function.
            self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_lens')
            self.seq_masks = tf.transpose(tf.sequence_mask(self.seq_lens, dtype=tf.float32), [1, 0])
            self.reset_state = tf.placeholder(dtype=tf.bool, shape=[], name='reset_state')
            if self.is_encoding:
                self.__build_encoding_model()
            else:
                self.__build_language_model()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
