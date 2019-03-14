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
BOU = '<U>'


def word_rep(word):
    lower = word.lower()
    if lower != word:
        return (BOU, unicodedata.normalize('NFD', lower))
    return (unicodedata.normalize('NFD', lower), )


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


def clean_text_v3(text, add_bos=True, add_eos=True):
    text = re.sub(r'[ ]*[\n\r]+[ ]*', ' _nl_ ', text)
    text = re.sub(r'[ ]+', '_sp_', text)
    text = re.sub(r'(\W)', r'_sp_\g<1>_sp_', text)
    text = re.sub(r'(_sp_)+', ' ', text)
    text = re.sub(r'\b\d+\b', '<number>', text)
    text = text.split() + [None]
    result = []
    count = 1
    for idx, word in enumerate(text[1:], start=1):
        if word == text[idx-1]:
            count += 1
        elif count > 1:
            result.append('<{}>'.format(count))
            result.extend(word_rep(text[idx-1]))
            count = 1
        else:
            result.extend(word_rep(text[idx-1]))
    if add_bos:
        result = [BOS] + result
    if add_eos:
        result = result + [EOS]
    return result


def clean_text_v4(text, add_eos=True):
    text = re.sub(r'[ ]*[\n\r]+[ ]*', ' _nl_ ', text)
    text = re.sub(r'[ ]+', '_sp_', text)
    text = re.sub(r'(\W)', r'_sp_\g<1>_sp_', text)
    text = re.sub(r'(_sp_)+', ' ', text)
    text = re.sub(r'\b\d+\b', '<number>', text)
    text = text.split() + [None]
    result = []
    count = 1
    for idx, word in enumerate(text[1:], start=1):
        if word == text[idx-1]:
            count += 1
        elif count > 1:
            count = count if count < 4 else 4
            result.append('<{}>'.format(count))
            result.extend(word_rep(text[idx-1]))
            count = 1
        else:
            result.extend(word_rep(text[idx-1]))
    if add_eos:
        result.append(EOS)
    return result


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
    lens = np.array([[len(y) for y in x] for x in seqs], dtype=np.int32)
    maxlens = np.max(lens)
    res = np.zeros(shape=(seqs.shape[0], seqs.shape[1], maxlens), dtype=np.int32)
    for ir in range(len(seqs)):
        for ic in range(len(seqs[ir])):
            s = seqs[ir][ic]
            res[ir][ic][:len(s)] = s
    return res, lens


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
        seq_len = get_random_bptt(bptt)
    seq_len = min(seq_len, len(source_word) - 1 - i)
    fw_data = source_char[i:i+seq_len]
    fw_data, fw_char_lens = pad_sequences(fw_data)
    fw_target = source_word[i+1:i+1+seq_len]
    bw_data = bw_source_char[i:i+seq_len]
    bw_data, bw_char_lens = pad_sequences(bw_data)
    bw_target = bw_source_word[i+1:i+1+seq_len]
    return (fw_data, fw_char_lens, fw_target), (bw_data, bw_char_lens, bw_target)


def get_random_bptt(bptt):
    real_bptt = bptt if np.random.random() < 0.95 else bptt / 2.
    # Prevent excessively small or negative sequence lengths
    real_bptt = min(max(5, int(np.random.normal(real_bptt, 5))), int(1.2*bptt))
    return real_bptt


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
    if logger.handlers:
        [handler.close() for handler in logger.handlers]
        logger.handlers = []
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def map_word_to_vector(new_w2i, old_w2i, old_matrix):
    nwords = len(new_w2i) + 1
    wdims = old_matrix.shape[-1]
    mean = old_matrix.mean(axis=0)
    std = old_matrix.std(axis=0)
    new_matrix = np.random.normal(loc=mean, scale=std, size=(nwords, wdims))
    for word, idx in new_w2i.items():
        if word in old_w2i:
            new_matrix[idx] = old_matrix[old_w2i[word]]
    return new_matrix


def combine_word2idx(old_w2i, new_w2i):
    result = {w: i for w, i in old_w2i.items()}
    start_i = max(result.values()) + 1
    for w in new_w2i:
        if w not in result:
            result[w] = start_i
            start_i += 1
    return result


def get_batch_classifier_and_tagger(texts, labels, tags, batch_size, splits, is_training=True):
    """texts is array of array of array of int"""
    """put sequences into buckets based on their lengths"""
    buckets = [[] for _ in range(len(splits))]
    for item, label, tag in zip(texts, labels, tags):
        ilen = len(item)
        for bucket, split in zip(buckets, splits[::-1]):
            if ilen >= split:
                bucket.append((item, label, tag))
                break
    if is_training:
        buckets = [np.random.permutation(x) for x in buckets]
        buckets = np.random.permutation(buckets)
    items = [x for y in buckets for x in y]
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_label = [x for _, x, _ in batch]
        batch = [x for x, _, _ in batch]
        tag = [x for _, _, x in batch]
        lens = np.array([len(x) for x in batch], dtype=np.int32)
        char_lens = [[len(w) for w in sent] for sent in batch]
        maxlens = np.max(lens)
        len_mat = np.zeros((len(batch), maxlens), dtype=np.int32)
        for r1, r2 in zip(len_mat, char_lens):
            r1[:len(r2)] = r2
        max_char_lens = np.max(len_mat)
        res_mat = np.zeros((len(batch), maxlens, max_char_lens), dtype=np.int32)
        for r1, r2 in zip(res_mat, batch):
            for c1, c2 in zip(r1[:len(r2)], r2):
                c1[:len(c2)] = c2
        tag_mat = np.zeros((len(batch), maxlens), dtype=np.int32)
        for r1, r2 in zip(tag_mat, tag):
            r1[:len(r2)] = r2
        yield np.transpose(res_mat, (1, 0, 2)), lens, np.transpose(len_mat, (1, 0)), batch_label, tag_mat


def get_batch_classifier_inference(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        lens = np.array([len(x) for x in batch], dtype=np.int32)
        char_lens = [[len(w) for w in sent] for sent in batch]
        maxlens = np.max(lens)
        len_mat = np.zeros((len(batch), maxlens), dtype=np.int32)
        for r1, r2 in zip(len_mat, char_lens):
            r1[:len(r2)] = r2
        max_char_lens = np.max(len_mat)
        res_mat = np.zeros((len(batch), maxlens, max_char_lens), dtype=np.int32)
        for r1, r2 in zip(res_mat, batch):
            for c1, c2 in zip(r1[:len(r2)], r2):
                c1[:len(c2)] = c2
        yield np.transpose(res_mat, (1, 0, 2)), lens, np.transpose(len_mat, (1, 0))


def slanted_triangular_learning_rate(total_iter, cut_frac, lr_max, ratio):
    curr_iter = 0
    cut = total_iter*cut_frac
    while curr_iter < total_iter:
        if curr_iter < cut:
            p = curr_iter/cut
        else:
            p = 1-(curr_iter-cut)/(cut*(1/cut_frac)-1)
        yield lr_max * (1+p*(ratio-1))/ratio
        curr_iter += 1


if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float)
    old_w2i = {'a': 1, 'b': 2, 'c': 3}
    new_w2i = {'c': 1, 'a': 2, 'b': 4, 'd': 3, 'e': 5}
    print(map_word_to_vector(new_w2i, old_w2i, a))
