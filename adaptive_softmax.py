# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""

import tensorflow as tf


def get_elements(data, indices):
    indeces = tf.range(0, tf.shape(indices)[0])*data.shape[1] + indices
    return tf.gather(tf.reshape(data, [-1]), indeces)


class AdaptiveSoftmaxLoss():
    def __init__(self, hidden_size, splits, name='AdaptiveSoftmaxLoss'):
        self.hidden_size = hidden_size
        self.splits = splits
        self.split_size = [splits[idx+1]-splits[idx]
                           for idx in range(len(splits)-1)]
        self.nsplits = len(self.split_size)
        self.name = name
        # Each of the splits that aren't in the head require a representative token, called the rep.
        # The probability given to this rep is the probability of selecting an item from the represented split
        if self.nsplits > 1:
            with tf.variable_scope(self.name):
                self.tail_vectors = tf.get_variable(name='tail_vectors', shape=(
                    hidden_size, self.nsplits - 1), initializer=tf.zeros_initializer)
                self.tail_biases = tf.get_variable(name='tail_biases', shape=(
                    self.nsplits - 1, ), initializer=tf.zeros_initializer)

    def split_on_targets(self, targets, hiddens):
        if self.nsplits == 1:
            return [targets], [hiddens]
        mask = tf.squeeze(tf.add_n([tf.to_int32(targets >= self.splits[idx])
                                    for idx in range(1, self.nsplits)]))
        return tf.dynamic_partition(targets, mask, self.nsplits), tf.dynamic_partition(hiddens, mask, self.nsplits)

    def apply(self, weight, bias, hiddens, targets, transpose=False):
        with tf.variable_scope(self.name):
            if transpose:
                weight = tf.transpose(weight, [1, 0])
            if len(targets.get_shape()) > 1:
                targets = tf.reshape(targets, shape=(-1,))
            if len(hiddens.get_shape()) > 2:
                hiddens = tf.reshape(
                    hiddens, shape=(-1, hiddens.get_shape()[-1]))
            split_targets, split_hiddens = self.split_on_targets(
                targets, hiddens)
            weights, biases = tf.split(
                weight, self.split_size, 1), tf.split(bias, self.split_size, 0)
            head_weight, head_bias = weights[0], biases[0]
            if self.nsplits > 1:
                head_weight = tf.concat([head_weight, self.tail_vectors], 1)
                head_bias = tf.concat([head_bias, self.tail_biases], 0)
            all_head_res = tf.nn.xw_plus_b(
                tf.concat(split_hiddens, 0), head_weight, head_bias)
            all_softmaxed_head_res = tf.nn.log_softmax(all_head_res)
            all_softmaxed_head_res = tf.split(
                all_softmaxed_head_res, [tf.shape(x)[0] for x in split_hiddens], axis=0)
            total_loss = - \
                tf.reduce_sum(get_elements(
                    all_softmaxed_head_res[0], split_targets[0]))
            idx = 1
            for tail_weight, tail_bias, hid, tar, softmaxed_head_res, start in zip(weights[1:], biases[1:], split_hiddens[1:], split_targets[1:], all_softmaxed_head_res[1:], self.splits[1:]):
                # Calculate the softmax for the words in the tombstone
                tail_res = tf.nn.xw_plus_b(hid, tail_weight, tail_bias)

                # Then we calculate p(tombstone) * p(word in tombstone)
                # Adding is equivalent to multiplication in log space
                head_entropy = tf.expand_dims(softmaxed_head_res[:, -idx], -1)
                tail_entropy = tf.nn.log_softmax(tail_res)
                entropy = head_entropy + tail_entropy
                total_loss -= tf.reduce_sum(get_elements(entropy, tar-start))
                idx += 1
            return total_loss / tf.to_float(tf.shape(targets)[0])


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)
    tf.set_random_seed(60)
    V = 8
    H = 10
    N = 100
    E = 10

    x = tf.to_int32(tf.floor(tf.random_uniform([N, 1], 0, 1) * 0.999 * V))
    y = tf.to_int32(tf.floor(tf.random_uniform([N, 1], 0, 1) * 0.999 * V))
    weight = tf.Variable(tf.random_normal([V, H], 0, 1))
    bias = tf.Variable(tf.ones([V]))
    embed = tf.nn.embedding_lookup(weight, y)
    loss = AdaptiveSoftmaxLoss(H, [0, 4, 8]).apply(
        weight, bias, embed, x, True)
    exp_loss = tf.exp(loss)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(1)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(E):
            _, loss_val, step = sess.run([train_op, exp_loss, global_step])
            print('Step', step, 'loss', loss_val)
