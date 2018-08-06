# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""

import tensorflow as tf


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
        pass
