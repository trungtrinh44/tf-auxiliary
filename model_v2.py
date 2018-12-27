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


def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
    carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
    transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
    return carry_gate * transform_gate + (1.0 - carry_gate) * x


def get_last_output(inputs, seq_lens):
    batch_size = tf.shape(inputs)[1]
    inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
    indices = (seq_lens-1)*batch_size + tf.range(0, batch_size, 1, dtype=tf.int32)
    indices = tf.maximum(0, indices)
    outputs = tf.gather(params=inputs, indices=indices)
    return outputs


class Embedding():
    def __init__(self, nwords, wdims, reuse, layers, n_highways, projection_dims, is_training, drop_e, name='word_embedding'):
        self.reuse = reuse
        self.layers = layers
        self.nhighways = n_highways
        self.projection_dims = projection_dims
        self.name = name
        self.is_training = is_training
        self.drop_e = drop_e
        self.nwords = nwords
        self.wdims = wdims

    def build(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.W = tf.get_variable(shape=[self.nwords, self.wdims], initializer=tf.glorot_uniform_initializer(), name="embedding_weight")
            self.conv = [tf.layers.Conv1D(num, fsz, padding='same', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer()) for fsz, num in self.layers]
            for conv in self.conv:
                conv.build((None, None, self.wdims))
            self.highweights = []
            self.nfilters = nfilters = sum(x for _, x in self.layers)
            self.output_shape = (None, None, nfilters)
            for i in range(self.nhighways):
                ww_carry = tf.get_variable(name='ww_carry_{}'.format(i), shape=(nfilters, nfilters), initializer=tf.glorot_uniform_initializer())
                bb_carry = tf.get_variable(name='bb_carry_{}'.format(i), shape=(nfilters,), initializer=tf.zeros_initializer())
                ww_tr = tf.get_variable(name='ww_tr_{}'.format(i), shape=(nfilters, nfilters), initializer=tf.glorot_uniform_initializer())
                bb_tr = tf.get_variable(name='bb_tr_{}'.format(i), shape=(nfilters,), initializer=tf.zeros_initializer())
                self.highweights.append((ww_carry, bb_carry, ww_tr, bb_tr))
            if isinstance(self.projection_dims, int) and self.projection_dims != nfilters:
                self.w_proj = tf.get_variable(name='w_proj', shape=(nfilters, self.projection_dims), initializer=tf.glorot_uniform_initializer())
                self.b_proj = tf.get_variable(name='b_proj', shape=(self.projection_dims,), initializer=tf.zeros_initializer())
                self.output_shape = (None, None, self.projection_dims)

    def call(self, inputs, char_lens):
        with tf.variable_scope(self.name, reuse=True):
            # Reshape from [T, B, C] to [T * B, C]
            s = tf.shape(inputs)
            T, B, max_char_lens = s[0], s[1], s[2]
            inputs = tf.reshape(inputs, (T * B, -1))
            char_lens = tf.reshape(char_lens, (T * B,))
            masks = tf.sequence_mask(char_lens, maxlen=max_char_lens, dtype=tf.float32)
            masks = tf.expand_dims(masks, axis=-1)
            with tf.device('/cpu:0'):
                if self.is_training and self.drop_e > 0.0:
                    W = embedding_dropout(self.W, dropout=self.drop_e)
                else:
                    W = self.W
                char_embed = tf.nn.embedding_lookup(W, inputs)
            conv_out = []
            char_embed *= masks
            for conv in self.conv:
                x = conv.call(char_embed)
                x = x * masks + (1.0 - masks) * -1e6
                x = tf.reduce_max(x, axis=1)
                conv_out.append(x)
            embedding = tf.concat(conv_out, axis=-1)
            for ww_carry, bb_carry, ww_tr, bb_tr in self.highweights:
                embedding = high(embedding, ww_carry, bb_carry, ww_tr, bb_tr)
            if isinstance(self.projection_dims, int) and self.projection_dims != self.nfilters:
                embedding = tf.matmul(embedding, self.w_proj) + self.b_proj
                embedding = tf.reshape(embedding, (T, B, self.projection_dims))
            else:
                embedding = tf.reshape(embedding, (T, B, self.nfilters))
            return embedding


class UniModel():
    def __init__(self, rnn_layers, projection_dims, skip_connection, is_training, fine_tune_lr, reuse, name):
        self.reuse = reuse
        self.name = name
        self.rnn_layers = rnn_layers
        self.projection_dims = projection_dims
        self.is_training = is_training
        self.skip_connection = skip_connection
        self.fine_tune_lr = fine_tune_lr

    def build(self, input_shape):
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.weights = []
            for idx, layer in enumerate(self.rnn_layers):
                cell = CudnnLSTM(num_layers=1, num_units=layer['units'], input_mode='linear_input', direction='unidirectional', dropout=0.0)
                cell.build(input_shape)
                weight = {'cell': cell}
                wdrop = layer.get('wdrop', 0.0)
                if self.is_training and wdrop > 0.0:
                    h_var_backup = tf.Variable(initial_value=tf.zeros(shape=[4*layer['units'], layer['units']]), trainable=False, name='h_var_backup_'+str(idx))
                    weight['h_var_backup'] = h_var_backup
                if isinstance(self.projection_dims, int) and self.projection_dims > 0:
                    w_proj = tf.get_variable(name='w_proj_{}'.format(idx), shape=(layer['units'], self.projection_dims), initializer=tf.glorot_uniform_initializer())
                    b_proj = tf.get_variable(name='b_proj_{}'.format(idx), shape=(self.projection_dims,), initializer=tf.zeros_initializer())
                    input_shape = (None, None, self.projection_dims)
                    weight['w_proj'] = w_proj
                    weight['b_proj'] = b_proj
                else:
                    input_shape = layer['units']
                self.weights.append(weight)

    def call(self, inputs, states):
        model = {}
        with tf.variable_scope(self.name, reuse=True):
            s = tf.shape(inputs)
            T, B = s[0], s[1]
            input_shape = (T, B, inputs.shape[-1])
            layer_outputs = [inputs]
            output_states = []
            for idx, (weight, l, state) in enumerate(zip(self.weights, self.rnn_layers, states)):
                cell = weight['cell']
                drop_i = l.get('drop_i', 0.0)
                if self.is_training and drop_i > 0.0:
                    inputs = tf.nn.dropout(x=inputs, keep_prob=1-drop_i, noise_shape=[1, input_shape[1], inputs.shape[-1]], name='drop_i_'+str(idx))
                wdrop = l.get('wdrop', 0.0)
                if self.is_training and wdrop > 0.0:
                    cell_var = cell.variables[0]
                    h_var_backup = weight['h_var_backup']
                    h_var = cell_var[inputs.shape[-1] * l['units'] * 4: -l['units'] * 8]
                    h_var = tf.reshape(h_var, [4*l['units'], l['units']]) + h_var_backup
                    keep_prob = 1-wdrop
                    random_tensor = keep_prob
                    random_tensor += tf.random_uniform([4*l['units'], 1], dtype=h_var.dtype)
                    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
                    binary_tensor = tf.floor(random_tensor)
                    new_h_var = tf.multiply(h_var, binary_tensor)
                    new_h_var = tf.reshape(new_h_var, [4*l['units']*l['units']])
                    h_var_backup = tf.assign(h_var_backup, tf.multiply(h_var, tf.subtract(1.0, binary_tensor)),
                                             validate_shape=True, use_locking=True, name='assign_h_var_backup_' + str(idx))
                    new_cell_var = tf.concat([cell_var[:inputs.shape[-1]*l['units']*4], new_h_var, cell_var[-l['units']*8:]], axis=0, name='new_cell_var_' + str(idx))
                    op = tf.assign(cell_var, new_cell_var, validate_shape=True, use_locking=True, name='assign_new_cell_var_' + str(idx))
                    with tf.control_dependencies([op, h_var_backup]):
                        outputs, new_state = cell.call(inputs=inputs, initial_state=state, training=self.is_training)
                else:
                    outputs, new_state = cell.call(inputs=inputs, initial_state=state, training=self.is_training)
                drop_o = l.get('drop_o', 0.0)
                if self.is_training and drop_o > 0.0:
                    outputs = tf.nn.dropout(x=outputs, keep_prob=1-drop_o, noise_shape=[1, input_shape[1], outputs.shape[-1]], name='drop_o_'+str(idx))
                if isinstance(self.projection_dims, int) and self.projection_dims > 0:
                    outputs = tf.reshape(outputs, (T * B, outputs.shape[-1]))
                    w_proj = weight['w_proj']
                    b_proj = weight['b_proj']
                    outputs = tf.matmul(outputs, w_proj) + b_proj
                    outputs = tf.reshape(outputs, (T, B, self.projection_dims))
                    if idx > 0 and self.skip_connection:
                        outputs = tf.add(outputs, inputs, name='skip_{}'.format(idx))
                if isinstance(self.fine_tune_lr, list):
                    outputs = apply_custom_lr(self.fine_tune_lr[idx])(outputs)
                layer_outputs.append(outputs)
                output_states.append(new_state)
                inputs = outputs
            model['layer_outputs'] = layer_outputs
            model['states'] = output_states
            return model


def build_uni_model_for_training(inputs, masks, share_W, share_b, reset_state, rnn_layers, projection_dims, skip_connection, fine_tune_lr, is_training, reuse, name):
    model = UniModel(rnn_layers, projection_dims, skip_connection, is_training, fine_tune_lr, reuse, name)
    model.build(inputs.shape)
    states = []
    saved_states = []
    batch_size = tf.shape(inputs)[1]
    with tf.variable_scope(name, reuse=reuse):
        for idx, layer in enumerate(rnn_layers):
            state = (tf.get_variable(shape=[1, 1, layer['units']], name='c_'+str(idx), trainable=False),
                     tf.get_variable(shape=[1, 1, layer['units']], name='h_'+str(idx), trainable=False))
            saved_states.append(state)
            for x in state:
                tf.add_to_collection(LSTM_SAVED_STATE, x)
            zeros = tf.zeros([1, batch_size, layer['units']], dtype=tf.float32)

            def if_true(): return (zeros, zeros)

            def if_false(): return state
            state = tf.cond(reset_state, if_true, if_false)
            states.append(state)
        model = model.call(inputs, states)
        ops = [tf.assign(s1, s2, validate_shape=False) for state_var, state_out in zip(saved_states, model['states']) for s1, s2 in zip(state_var, state_out)]
        ops = tf.group(ops)
        with tf.control_dependencies([ops]):
            rnn_outputs = tf.multiply(model['layer_outputs'][-1], masks, name='rnn_outputs')
            decoder = tf.nn.xw_plus_b(tf.reshape(rnn_outputs, (-1, rnn_outputs.shape[-1])), share_W, share_b)
            decoder = tf.reshape(decoder, (-1, batch_size, share_W.shape[-1]))
        model['decoder'] = decoder
        model['rnn_outputs'] = rnn_outputs
    return model


def build_word_embedding_for_training(inputs, char_lens, nwords, wdims, reuse, layers, nhighways, projection_dims, is_training, drop_e, name='word_embedding'):
    embedding = Embedding(nwords, wdims, reuse, layers, nhighways, projection_dims, is_training, drop_e, name)
    embedding.build()
    return embedding.call(inputs, char_lens)


class LanguageModel():
    def __init__(self, char_vocab_size, char_vec_size, char_cnn_options, vocab_size, rnn_layers, drop_e, is_training, projection_dims,
                 skip_connection, fine_tune_lr=None, is_encoding=False, custom_getter=None, reuse=False, name='LanguageModel'):
        self.vocab_size = vocab_size
        self.char_vocab_size = char_vocab_size
        self.rnn_layers = rnn_layers
        self.drop_e = drop_e
        self.name = name
        self.is_training = is_training
        self.custom_getter = custom_getter
        self.reuse = reuse
        self.fine_tune_lr = fine_tune_lr[::-1] if isinstance(fine_tune_lr, list) else None
        self.char_cnn_options = char_cnn_options
        self.char_vec_size = char_vec_size
        self.is_encoding = is_encoding
        self.projection_dims = projection_dims
        self.skip_connection = skip_connection

    def build_language_model(self):
        self.fw_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='fw_inputs')
        self.bw_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='bw_inputs')
        self.reset_state = tf.placeholder(dtype=tf.bool, shape=[], name='reset_state')
        self.fw_char_lens = tf.placeholder(dtype=tf.int32, shape=[None, None], name='fw_char_lens')
        self.bw_char_lens = tf.placeholder(dtype=tf.int32, shape=[None, None], name='bw_char_lens')
        seq_masks = tf.expand_dims(self.seq_masks, axis=-1)
        self.share_decode_W = tf.get_variable(
            name='decode_W',
            shape=(self.projection_dims if isinstance(self.projection_dims, int) and self.projection_dims > 0 else self.rnn_layers[-1]['units'], self.vocab_size),
            initializer=tf.glorot_uniform_initializer()
        )
        self.share_decode_b = tf.get_variable(name='decode_b', shape=(self.vocab_size,), initializer=tf.zeros_initializer())
        fw_emb = build_word_embedding_for_training(self.fw_inputs, self.fw_char_lens, self.char_vocab_size, self.char_vec_size, self.reuse,
                                                   self.char_cnn_options['layers'], self.char_cnn_options['n_highways'], self.projection_dims, self.is_training, self.drop_e)
        bw_emb = build_word_embedding_for_training(self.bw_inputs, self.bw_char_lens, self.char_vocab_size, self.char_vec_size, True,
                                                   self.char_cnn_options['layers'], self.char_cnn_options['n_highways'], self.projection_dims, self.is_training, self.drop_e)

        if isinstance(self.fine_tune_lr, list):
            custom_lr = apply_custom_lr(self.fine_tune_lr[0])
            fw_emb = custom_lr(fw_emb)
            bw_emb = custom_lr(bw_emb)
        self.fw_model = build_uni_model_for_training(
            fw_emb, seq_masks, self.share_decode_W, self.share_decode_b, self.reset_state, self.rnn_layers, self.projection_dims, self.skip_connection,
            self.fine_tune_lr[1:] if isinstance(self.fine_tune_lr, list) else None, self.is_training, self.reuse, 'LMFW'
        )
        self.bw_model = build_uni_model_for_training(
            bw_emb, seq_masks, self.share_decode_W, self.share_decode_b, self.reset_state, self.rnn_layers, self.projection_dims, self.skip_connection,
            self.fine_tune_lr[1:] if isinstance(self.fine_tune_lr, list) else None, self.is_training, self.reuse, 'LMBW'
        )

    def build_encoding_model(self):
        self.fw_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name='fw_inputs')
        self.bw_inputs = tf.reverse_sequence(input=self.fw_inputs, seq_lengths=self.seq_lens, seq_axis=0, batch_axis=1, name='bw_inputs')
        self.fw_char_lens = tf.placeholder(dtype=tf.int32, shape=[None, None], name='fw_char_lens')
        self.bw_char_lens = tf.reverse_sequence(input=self.fw_char_lens, seq_lengths=self.seq_lens, seq_axis=0, batch_axis=1, name='bw_char_lens')
        self.inputs = self.fw_inputs
        self.char_lens = self.fw_char_lens
        self.bptt = tf.placeholder(dtype=tf.int32, name='bptt', shape=())
        # seq_masks = tf.expand_dims(self.seq_masks, axis=-1)
        input_shape = tf.shape(self.inputs)
        B = input_shape[1]
        fw_model = UniModel(self.rnn_layers, self.projection_dims, self.skip_connection, self.is_training, self.fine_tune_lr[1:] if isinstance(self.fine_tune_lr, list) else None, self.reuse, 'LMFW')
        bw_model = UniModel(self.rnn_layers, self.projection_dims, self.skip_connection, self.is_training, self.fine_tune_lr[1:] if isinstance(self.fine_tune_lr, list) else None, self.reuse, 'LMBW')
        embed_model = Embedding(self.char_vocab_size, self.char_vec_size, self.reuse, self.char_cnn_options['layers'],
                                self.char_cnn_options['n_highways'], self.projection_dims, self.is_training, self.drop_e)
        embed_model.build()
        if isinstance(self.fine_tune_lr, list):
            embed_custom_lr = apply_custom_lr(self.fine_tune_lr[0])
        else:
            embed_custom_lr = lambda x: x
        fw_model.build(embed_model.output_shape)
        bw_model.build(embed_model.output_shape)
        initial_states = []
        start_max_vals = []
        start_mean_vals = []
        start_outputs = []
        start_last_outputs = []
        start_output_shapes = []
        projection_dims = self.projection_dims if isinstance(self.projection_dims, int) and self.projection_dims > 0 else None
        for layer in self.rnn_layers:
            zeros = tf.fill(value=0.0, dims=(1, B, layer['units']))
            initial_states.append((zeros, zeros))
            dims = projection_dims if self.projection_dims else layer['units']
            max_val = tf.fill(value=-1e6, dims=(B, dims))
            mean_val = tf.fill(value=0.0, dims=(B, dims))
            start_output = tf.fill(value=0.0, dims=(0, B, dims))
            start_max_vals.append(max_val)
            start_mean_vals.append(mean_val)
            start_last_outputs.append(mean_val)
            start_outputs.append(start_output)
            start_output_shapes.append(tf.TensorShape((None, None, dims)))
        max_len = tf.reduce_max(self.seq_lens)

        def cond(i, state, max_vals, mean_vals, all_outputs, last_outputs): return i < max_len

        def body(embed, model, inputs, char_lens, sl, bptt, max_len):
            def child(i, state, max_vals, mean_vals, all_outputs, last_outputs):
                i_to = tf.minimum(i+bptt, max_len)
                slice_inputs = inputs[i:i_to]
                slice_char_lens = char_lens[i:i_to]
                slice_inputs = embed.call(slice_inputs, slice_char_lens)
                slice_inputs = embed_custom_lr(slice_inputs)
                output_dict = model.call(slice_inputs, state)
                slice_seq_lens = tf.minimum(sl-i, bptt)
                mask = tf.expand_dims(tf.transpose(tf.sequence_mask(slice_seq_lens, dtype=tf.float32), (1, 0)), axis=-1)
                next_max_vals = []
                next_mean_vals = []
                new_all_outputs = []
                new_last_outputs = []
                for max_val, mean_val, outputs, past_outputs, last_output in zip(max_vals, mean_vals, output_dict['layer_outputs'][1:], all_outputs, last_outputs):
                    max_outputs = outputs * mask + (1 - mask) * -1e6
                    max_val = tf.maximum(max_val, tf.reduce_max(max_outputs, axis=0))
                    mean_outputs = outputs * mask
                    mean_val = (mean_val * tf.expand_dims(tf.to_float(tf.minimum(i, sl)), axis=-1) + tf.reduce_sum(mean_outputs, axis=0)) / tf.expand_dims(tf.to_float(tf.minimum(i_to, sl)), axis=-1)
                    next_max_vals.append(max_val)
                    next_mean_vals.append(mean_val)
                    new_all_outputs.append(tf.concat((past_outputs, mean_outputs), axis=0))
                    last_val = get_last_output(mean_outputs, slice_seq_lens)
                    last_val = tf.where(slice_seq_lens > 0, last_val, last_output)
                    new_last_outputs.append(last_val)
                return i_to, output_dict['states'], next_max_vals, next_mean_vals, new_all_outputs, new_last_outputs
            return child
        start_i = tf.constant(0, dtype=tf.int32, shape=(), name='start_i')
        _, _, fw_layerwise_max, fw_layerwise_avg, fw_outputs, fw_last_output = tf.while_loop(cond, body(embed_model, fw_model, self.fw_inputs, self.fw_char_lens, self.seq_lens, self.bptt, max_len),
                                                                                             [start_i, initial_states, start_max_vals, start_mean_vals, start_outputs, start_last_outputs],
                                                                                             [start_i.get_shape(), [(x.get_shape(), y.get_shape()) for x, y in initial_states],
                                                                                              [x.get_shape() for x in start_max_vals], [x.get_shape() for x in start_mean_vals], start_output_shapes,
                                                                                              [x.get_shape() for x in start_last_outputs]])
        _, _, bw_layerwise_max, bw_layerwise_avg, bw_outputs, bw_last_output = tf.while_loop(cond, body(embed_model, bw_model, self.bw_inputs, self.bw_char_lens, self.seq_lens, self.bptt, max_len),
                                                                                             [start_i, initial_states, start_max_vals, start_mean_vals, start_outputs, start_last_outputs],
                                                                                             [start_i.get_shape(), [(x.get_shape(), y.get_shape()) for x, y in initial_states],
                                                                                              [x.get_shape() for x in start_max_vals], [x.get_shape() for x in start_mean_vals], start_output_shapes,
                                                                                              [x.get_shape() for x in start_last_outputs]])
        self.layerwise_max = [tf.concat((fw, bw), axis=-1) for fw, bw in zip(fw_layerwise_max, bw_layerwise_max)]
        self.layerwise_avg = [tf.concat((fw, bw), axis=-1) for fw, bw in zip(fw_layerwise_avg, bw_layerwise_avg)]
        self.layerwise_last = [tf.concat((fw, bw), axis=-1) for fw, bw in zip(fw_last_output, bw_last_output)]
        self.timewise_outputs = [tf.concat((fw, tf.reverse_sequence(input=bw, seq_lengths=self.seq_lens, seq_axis=0, batch_axis=1)), axis=-1) for fw, bw in zip(fw_outputs, bw_outputs)]
        self.layerwise_encode = [tf.concat(out, axis=-1) for out in zip(fw_layerwise_avg, fw_layerwise_max, bw_layerwise_avg, bw_layerwise_max)]

    def build_model(self):
        with tf.variable_scope(self.name, custom_getter=self.custom_getter, reuse=self.reuse):
            # Inputs must be sequences of token ids with shape [time, batch, depth]
            # rnn_layers is a list of dictionaries, each contains all the parameters of the __get_rnn_cell function.
            self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_lens')
            self.seq_masks = tf.transpose(tf.sequence_mask(self.seq_lens, dtype=tf.float32), [1, 0])
            if self.is_encoding:
                self.build_encoding_model()
            else:
                self.build_language_model()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Classifier():
    def __init__(self, layers, n_classes, is_training, reuse, custom_getter=None, name='Classifier'):
        self.layers = layers
        self.n_classes = n_classes
        self.is_training = is_training
        self.reuse = reuse
        self.name = name
        self.custom_getter = custom_getter

    def build(self, inputs):
        outputs = inputs
        with tf.variable_scope(self.name, custom_getter=self.custom_getter, reuse=self.reuse):
            layer = self.layers[0]
            if layer.get('batch_norm', False):
                outputs = tf.layers.batch_normalization(outputs, trainable=self.is_training, training=self.is_training)
            if self.is_training and layer.get('drop_out', False):
                outputs = tf.layers.dropout(outputs, rate=layer.get('drop_out'))
            for idx, layer in enumerate(self.layers[1:], 1):
                outputs = tf.layers.dense(outputs, layer['units'], kernel_initializer=tf.glorot_uniform_initializer(), name='layer_{}'.format(idx))
                if layer.get('batch_norm', False):
                    outputs = tf.layers.batch_normalization(outputs, trainable=self.is_training, training=self.is_training)
                if self.is_training and layer.get('drop_out', False):
                    outputs = tf.layers.dropout(outputs, rate=layer.get('drop_out'))
                activation = layer.get('activation', 'linear')
                if activation == 'tanh':
                    outputs = tf.tanh(outputs)
                elif activation == 'sigmoid':
                    outputs = tf.sigmoid(outputs)
                elif activation == 'relu':
                    outputs = tf.nn.relu(outputs)
            self.logits = tf.layers.dense(outputs, self.n_classes, kernel_initializer=tf.glorot_uniform_initializer(), name='logits')
            self.probs = tf.nn.softmax(self.logits)

def build_lm_classifier_inference(lm_params, cls_params):
    lm = LanguageModel(**lm_params, is_training=False, is_encoding=True)
    classifier = Classifier(**cls_params, is_training=False)
    lm.build_model()
    classifier.build(lm.layerwise_encode[-1])
    return lm, classifier