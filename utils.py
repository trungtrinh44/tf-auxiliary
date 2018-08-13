# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import logging

import numpy as np
import tensorflow as tf


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(
        ':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars, max_to_keep=1)
    saver.restore(session, save_file)
    return restore_vars, saver


def get_batch(source, bptt, i, evaluate=False):
    if evaluate:
        seq_len = bptt
    else:
        real_bptt = bptt if np.random.random() < 0.95 else bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = min(
            max(5, int(np.random.normal(real_bptt, 5))), int(1.2*bptt))
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target


def batchify(source, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(source) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = source[:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    data = np.reshape(data, [bsz, nbatch])
    return data


def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger
