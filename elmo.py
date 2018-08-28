# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:44:35 2018

@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import tensorflow as tf

def elmo_embedding(inputs, seq_lens, l2_coef=1e-3, layer_norm=False, name='elmo_embedding', reuse=False):
    """
    Inputs have shape [Time, Batch, Dimension, Layer]
    """
    with tf.variable_scope(name, reuse=reuse):
        l2_reg = 0.0
        W = tf.get_variable(name='W', shape=(inputs.shape[3], ), initializer=tf.zeros_initializer(), trainable=True)
        if l2_coef > 0.0:
            l2_reg = tf.nn.l2_loss(W) * l2_coef
        normalize_W = tf.nn.softmax(W, axis=-1, name='normalize_W')
        normalize_W = tf.unstack(normalize_W, inputs.shape[3], axis=0)
        layers = tf.unstack(inputs, inputs.shape[3], axis=3)
        print(layers)
        if layer_norm:
            mask_float = tf.sequence_mask(seq_lens, dtype=tf.float32)
            broadcast_mask = tf.expand_dims(mask_float, axis=-1)
            float_lens = tf.to_float(seq_lens)
            lm_dim = tf.to_float(inputs.shape[2])
            def _do_ln(x):
                # do layer normalization excluding the mask
                x_masked = x * broadcast_mask
                N = tf.reshape(float_lens * lm_dim, (-1, 1, 1))
                mean = tf.reduce_sum(x_masked, axis=(1, 2), keepdims=True) / N
                variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask)**2, axis=(1, 2), keepdims=True) / N
                return tf.nn.batch_normalization(
                    x, mean, variance, None, None, 1e-12
                )
            layers = [_do_ln(l) for l in layers]
        result = tf.add_n([w * l for w, l in zip(normalize_W, layers)])
        gamma = tf.get_variable(name='gamma', shape=(1,), initializer=tf.ones_initializer(), trainable=True)
        result = result * gamma
    return result, l2_reg