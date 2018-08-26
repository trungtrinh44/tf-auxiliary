# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:44:35 2018

@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import tensorflow as tf

def elmo_embedding(inputs, l2_coef=1e-3, layer_norm=False, name='elmo_embedding', reuse=False):
    """
    Inputs have shape [Time, Batch, Dimension, Layer]
    """
    with tf.variable_scope(name, reuse=reuse):
        l2_reg = 0.0
        W = tf.get_variable(name='W', shape=(1,1,1,inputs.shape[3]), initializer=tf.zeros_initializer(), trainable=True)
        if l2_coef > 0.0:
            l2_reg = tf.nn.l2_loss(W) * l2_coef
        normalize_W = tf.nn.softmax(W, axis=-1, name='normalize_W')
        result = inputs * normalize_W
        result = tf.reduce_sum(result, axis=-1, keepdims=False)
        gamma = tf.get_variable(name='gamma', shape=(1,), initializer=tf.ones_initializer(), trainable=True)
        result = reuse * gamma
    return result, l2_reg