# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""

import tensorflow as tf


def get_elements(data, indices):
    indeces = tf.range(0, tf.shape(indices)[0])*data.shape[1] + indices
    return tf.gather(tf.reshape(data, [-1]), indeces)


class SplitCrossEntropyLoss():
    def __init__(self, hidden_size, splits, name='SplitCrossEntropyLoss'):
        self.hidden_size = hidden_size
        self.splits = [0] + splits + [9223372036854775807]
        self.nsplits = len(self.splits) - 1
        self.name = name
        # Each of the splits that aren't in the head require a representative token, called the rep.
        # The probability given to this rep is the probability of selecting an item from the represented split
        if self.nsplits > 1:
            with tf.variable_scope(self.name):
                self.tail_vectors = tf.get_variable(name='tail_vectors', shape=(
                    self.nsplits - 1, hidden_size), initializer=tf.zeros_initializer)
                self.tail_biases = tf.get_variable(name='tail_biases', shape=(
                    self.nsplits - 1, ), initializer=tf.zeros_initializer)

    def split_on_targets(self, targets, hiddens):
        if self.nsplits == 1:
            return [targets], [hiddens]
        mask = tf.squeeze(tf.add_n([tf.to_int32(targets >= self.splits[idx])
                                    for idx in range(1, self.nsplits)]))
        return tf.dynamic_partition(targets, mask, self.nsplits), tf.dynamic_partition(hiddens, mask, self.nsplits)

    def apply(self, weight, bias, hiddens, targets):
        with tf.variable_scope(self.name):
            if len(targets.get_shape()) > 1:
                targets = tf.reshape(targets, shape=(-1,))
            if len(hiddens.get_shape()) > 2:
                hiddens = tf.reshape(
                    hiddens, shape=(-1, hiddens.get_shape()[-1]))
            split_targets, split_hiddens = self.split_on_targets(
                targets, hiddens)
            start, end = self.splits[0], self.splits[1]
            if end - start:
                head_weight, head_bias = weight[start:end], bias[start:end]
            else:
                head_weight, head_bias = None, None
            if self.nsplits > 1:
                head_weight = tf.concat(
                    [head_weight, self.tail_vectors], 0) if head_weight is not None else self.tail_vectors
                head_weight = tf.transpose(head_weight, [1, 0])
                head_bias = tf.concat(
                    [head_bias, self.tail_biases], 0) if head_bias is not None else self.tail_biases
            head_res = tf.nn.xw_plus_b(
                split_hiddens[0], head_weight, head_bias)
            softmaxed_head_res = tf.nn.log_softmax(head_res)
            total_loss = - \
                tf.reduce_sum(get_elements(
                    softmaxed_head_res, split_targets[0]))
            for idx, (hid, tar) in enumerate(zip(split_hiddens[1:], split_targets[1:]), 1):
                head_res = tf.nn.xw_plus_b(hid, head_weight, head_bias)
                softmaxed_head_res = tf.nn.log_softmax(head_res)
                start, end = self.splits[idx], self.splits[idx + 1]
                tail_weight = weight[start:end]
                tail_bias = bias[start:end]

                # Calculate the softmax for the words in the tombstone
                tail_res = tf.nn.xw_plus_b(
                    hid, tf.transpose(tail_weight, [1, 0]), tail_bias)

                # Then we calculate p(tombstone) * p(word in tombstone)
                # Adding is equivalent to multiplication in log space
                head_entropy = tf.expand_dims(softmaxed_head_res[:, -idx], -1)
                tail_entropy = tf.nn.log_softmax(tail_res)
                entropy = head_entropy + tail_entropy
                total_loss -= tf.reduce_sum(get_elements(entropy, tar-start))
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
    loss = SplitCrossEntropyLoss(H, [V//2]).apply(weight, bias, embed, x)
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
