# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import logging
import re
import unicodedata

import numpy as np
import tensorflow as tf

BOS = '<S>'
EOS = '</S>'


def clean_text(text, add_bos=True, add_eos=True):
    text = re.sub(r'[ ]*[\n\r]+[ ]*', ' _nl_ ', text)
    text = re.sub(r'[ ]+', '_sp_', text)
    text = re.sub(r'(\W)', r'_sp_\g<1>_sp_', text)
    text = re.sub(r'(_sp_)+', ' ', text)
    text = re.sub(r'\b\d+\b', '<number>', text)
    if add_bos:
        text = BOS + ' ' + text
    if add_eos:
        text = text + ' ' + EOS
    return text


def clean_text_v2(text, add_bos=True, add_eos=True):
    text = re.sub(r'[ ]*[\n\r]+[ ]*', ' _nl_ ', text)
    text = re.sub(r'[ ]+', '_sp_', text)
    text = re.sub(r'(\W)', r'_sp_\g<1>_sp_', text)
    text = re.sub(r'(_sp_)+', ' ', text)
    text = re.sub(r'\b([^\d]*)\d+([^\d]*)\b', r'\g<1> <number> \g<2>', text)
    if add_bos:
        text = BOS + ' ' + text
    if add_eos:
        text = text + ' ' + EOS
    return text


def pad_sequences(seqs):
    maxlens = max(len(y) for x in seqs for y in x)
    res = np.zeros(
        shape=(seqs.shape[0], seqs.shape[1], maxlens), dtype=np.int32)
    for ir in range(len(seqs)):
        for ic in range(len(seqs[ir])):
            s = seqs[ir][ic]
            res[ir][ic][:len(s)] = s
    return res


def optimistic_restore(session, variables, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], variables), variables))
    with tf.variable_scope('', reuse=True):
        for _, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars, max_to_keep=1)
    saver.restore(session, save_file)
    return restore_vars, saver


def get_batch(source_word, source_char, bptt, i, evaluate=False):
    bw_source_word = np.flip(source_word, axis=0)
    bw_source_char = np.flip(source_char, axis=0)
    if evaluate:
        seq_len = bptt
    else:
        real_bptt = bptt if np.random.random() < 0.95 else bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = min(
            max(5, int(np.random.normal(real_bptt, 5))), int(1.2*bptt))
    seq_len = min(seq_len, len(source_word) - 1 - i)
    fw_data = source_char[i:i+seq_len]
    fw_data = pad_sequences(fw_data)
    fw_target = source_word[i+1:i+1+seq_len]
    bw_data = bw_source_char[i:i+seq_len]
    bw_data = pad_sequences(bw_data)
    bw_target = bw_source_word[i+1:i+1+seq_len]
    return (fw_data, fw_target), (bw_data, bw_target)


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


def map_word_to_vector(new_w2i, old_w2i, old_matrix):
    nwords = len(new_w2i)
    wdims = old_matrix.shape[-1]
    mean = old_matrix.mean(axis=0)
    std = old_matrix.std(axis=0)
    new_matrix = np.random.normal(loc=mean, scale=std, size=(nwords, wdims))
    for word, idx in new_w2i.items():
        if word in old_w2i:
            new_matrix[idx-1] = old_matrix[old_w2i[word]-1]
    return new_matrix


if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float)
    old_w2i = {'a': 1, 'b': 2, 'c': 3}
    new_w2i = {'c': 1, 'a': 2, 'b': 4, 'd': 3, 'e': 5}
    print(map_word_to_vector(new_w2i, old_w2i, a))
