# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import json
import os
import shutil
import time
import types

import numpy as np
import tensorflow as tf

from model_v2 import (LSTM_SAVED_STATE, Classifier, LanguageModel,
                      SequenceTagger)
from utils import (get_batch, get_batch_classifier_and_tagger, get_getter, get_logger,
                   get_random_bptt)

name2optimizer = {
    'adam': tf.train.AdamOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
    'momentum': tf.train.MomentumOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'adamw': tf.contrib.opt.AdamWOptimizer
}


class Trainer():
    def __init__(self, model_configs, optimizer, wdecay, alpha, beta, bptt, negative_samples, log_path, train_summary_dir,
                 test_summary_dir, checkpoint_dir, save_freq, clip_norm=None, clip_max=1.0, clip_min=-1.0, use_ema=False, ema_decay=0.998, fine_tune=False, name='LM_Trainer'):
        self.model_configs = model_configs
        self.optimizer = optimizer
        self.name = name
        self.train_summary_dir = train_summary_dir
        self.test_summary_dir = test_summary_dir
        self.checkpoint_dir = checkpoint_dir
        self.alpha = alpha
        self.beta = beta
        self.bptt = bptt
        self.clip_norm = clip_norm
        self.clip_max = clip_max
        self.clip_min = clip_min
        os.makedirs(os.path.join(self.checkpoint_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, 'test'), exist_ok=True)
        self.logger = get_logger(log_path)
        self.use_ema = use_ema
        self.save_freq = save_freq
        self.wdecay = wdecay
        self.ema_decay = ema_decay
        self.negative_samples = negative_samples
        self.fine_tune = fine_tune

    def save_configs(self):
        with open(os.path.join(self.checkpoint_dir, 'model_configs.json'), 'w') as out:
            json.dump(self.model_configs, out)

    def build_classifier_and_sequence_tagger(self, classifier_configs, tagger_configs, folder_name, save_optimizer_var=True):
        # Ad hoc function for fast experiment. Use for intent classification and slot tagging in chatbot.
        with open(os.path.join(self.checkpoint_dir, 'classifier_configs.json'), 'w') as out:
            json.dump(classifier_configs, out)
        with open(os.path.join(self.checkpoint_dir, 'tagger_configs.json'), 'w') as out:
            json.dump(tagger_configs, out)
        self.classifier_configs = classifier_configs
        self.tagger_configs = tagger_configs
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        self.session = tf.Session(config=config)
        if self.fine_tune:
            self.fine_tune_rate = [tf.placeholder(dtype=tf.float32, name='lr_rate_{}'.format(i), shape=()) for i in range(len(self.model_configs['rnn_layers'])+1)]
        else:
            self.fine_tune_rate = None
        self.model_train = LanguageModel(**self.model_configs, reuse=False, is_training=True, fine_tune_lr=self.fine_tune_rate, is_encoding=True, is_cpu=False)  # Only train on GPU for now
        self.model_train.build_model()
        self.train_classifier = Classifier(**classifier_configs, is_training=True, reuse=False)
        self.train_classifier.build(self.model_train.layerwise_encode[-1])
        self.train_tagger = SequenceTagger(**tagger_configs, is_training=True, reuse=False)
        self.train_tagger.build(tf.transpose(self.model_train.timewise_outputs[-1], (1, 0, 2)), self.model_train.seq_lens)
        with tf.variable_scope(self.name):
            self.true_y = tf.placeholder(dtype=tf.int32, shape=[None], name='true_y')
            self.true_seq = tf.placeholder(dtype=tf.int32, shape=(None, None), name='true_seq')
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.true_y, logits=self.train_classifier.logits)  # classifier loss
            tagger_loss, _ = tf.contrib.crf.crf_log_likelihood(inputs=self.train_tagger.logits, tag_indices=self.true_seq,
                                                               sequence_lengths=self.model_train.seq_lens, transition_params=self.train_tagger.transition_params)
            self.loss -= tagger_loss
            self.loss = tf.reduce_mean(self.loss)
            self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
            self.train_class_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.train_classifier.logits, axis=-1, output_type=tf.int32), self.true_y)))
            self.train_tag_acc = tf.reduce_sum(tf.to_float(tf.equal(self.train_tagger.decode_sequence, self.true_seq)) *
                                               self.model_train.orig_seq_masks)/tf.reduce_sum(tf.to_float(self.model_train.seq_lens))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = name2optimizer[self.optimizer['name']](**self.optimizer['params'], learning_rate=self.lr) if isinstance(self.optimizer, dict) else self.optimizer(self.lr)
        self.grads, self.vars = zip(*self.optimizer.compute_gradients(self.loss))
        if isinstance(self.clip_norm, float):
            self.grads, _ = tf.clip_by_global_norm(self.grads, clip_norm=self.clip_norm)
        self.grads = [tf.clip_by_value(grad, self.clip_min, self.clip_max) for grad in self.grads]  # Clip gradient between -1.0 and 1.0 just to be safe
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.vars), global_step=self.global_step)
        # Add summary op
        train_summaries = [tf.summary.scalar('Loss', self.loss), tf.summary.scalar('Learning_rate', self.lr)]
        self.train_summaries = tf.summary.merge(train_summaries, name='train_summaries')
        if self.use_ema:
            ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay, num_updates=self.global_step)
            var_class = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_train.name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                                                       self.train_classifier.name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.train_tagger.name)
            with tf.control_dependencies([self.train_op]):
                self.train_op = ema.apply(var_class)
            self.model_test = LanguageModel(**self.model_configs, reuse=True, is_training=False, custom_getter=get_getter(ema), name=self.model_train.name, is_encoding=True, is_cpu=False)
            self.test_classifier = Classifier(**classifier_configs, is_training=False, reuse=True, custom_getter=get_getter(ema), name=self.train_classifier.name)
            self.test_tagger = SequenceTagger(**tagger_configs, is_training=False, reuse=True, custom_getter=get_getter(ema), name=self.train_tagger.name)
            self.test_saver = tf.train.Saver({v.op.name: ema.average(v) for v in var_class}, max_to_keep=100)
            self.ema = ema
        else:
            var_class = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_train.name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                                                       self.train_classifier.name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.train_tagger.name)
            self.model_test = LanguageModel(**self.model_configs, reuse=True, is_training=False, custom_getter=None, name=self.model_train.name, is_encoding=True, is_cpu=False)
            self.test_classifier = Classifier(**classifier_configs, is_training=False, reuse=True, custom_getter=None, name=self.train_classifier.name)
            self.test_tagger = SequenceTagger(**tagger_configs, is_training=False, reuse=True, custom_getter=None, name=self.train_tagger.name)
            self.test_saver = tf.train.Saver(var_class, max_to_keep=100)
        self.model_test.build_model()
        self.test_classifier.build(self.model_test.layerwise_encode[-1])
        self.test_tagger.build(tf.transpose(self.model_test.timewise_outputs[-1], (1, 0, 2)), self.model_test.seq_lens)
        self.test_loss = tf.reduce_mean(tf.subtract(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.true_y, logits=self.test_classifier.logits),
            tf.contrib.crf.crf_log_likelihood(inputs=self.test_tagger.logits, tag_indices=self.true_seq,
                                              sequence_lengths=self.model_test.seq_lens, transition_params=self.test_tagger.transition_params)[0]
        ))
        self.test_class_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.test_classifier.logits, axis=-1, output_type=tf.int32), self.true_y)))
        self.test_tag_acc = tf.reduce_sum(tf.to_float(tf.equal(self.test_tagger.decode_sequence, self.true_seq)) * self.model_test.orig_seq_masks)/tf.reduce_sum(tf.to_float(self.model_test.seq_lens))
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.checkpoint_dir, folder_name))
        self.session.run(tf.global_variables_initializer())
        black_list_var = tf.get_collection(LSTM_SAVED_STATE)
        if not save_optimizer_var:
            black_list_var.extend(self.optimizer.variables())
        self.train_saver = tf.train.Saver([x for x in tf.global_variables() if x not in black_list_var], max_to_keep=2)
        if latest_checkpoint is not None:
            self.train_saver.restore(self.session, latest_checkpoint)

    def restore_language_model(self, checkpoint_path):
        var_class = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_train.name)
        if self.use_ema:
            var_class += [self.ema.average(x) for x in var_class]
        self.language_model_saver = tf.train.Saver(var_class, max_to_keep=100)
        self.language_model_saver.restore(self.session, checkpoint_path)

    def build(self, folder_name='train', save_optimizer_var=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        self.session = tf.Session(config=config)
        if self.fine_tune:
            self.fine_tune_rate = [tf.placeholder(dtype=tf.float32, name='lr_rate_{}'.format(i), shape=()) for i in range(len(self.model_configs['rnn_layers'])+1)]
        else:
            self.fine_tune_rate = None
        self.model_train = LanguageModel(**self.model_configs, reuse=False, is_training=True, fine_tune_lr=self.fine_tune_rate, is_cpu=False)
        self.model_train.build_model()
        with tf.variable_scope(self.name):
            self.fw_y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='fw_y')
            self.bw_y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='bw_y')
            self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
            self.fw_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=tf.transpose(self.model_train.share_decode_W, (1, 0)),
                biases=self.model_train.share_decode_b,
                labels=tf.reshape(self.fw_y, [-1, 1]),
                inputs=tf.reshape(self.model_train.fw_model['rnn_outputs'], (-1, self.model_train.fw_model['rnn_outputs'].shape[-1])),
                num_sampled=self.negative_samples,
                num_classes=self.model_configs['vocab_size'],
                num_true=1,
                remove_accidental_hits=True,
                partition_strategy='div',
                name='fw_loss'
            ))
            self.bw_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=tf.transpose(self.model_train.share_decode_W, (1, 0)),
                biases=self.model_train.share_decode_b,
                labels=tf.reshape(self.bw_y, [-1, 1]),
                inputs=tf.reshape(self.model_train.bw_model['rnn_outputs'], (-1, self.model_train.bw_model['rnn_outputs'].shape[-1])),
                num_sampled=self.negative_samples,
                num_classes=self.model_configs['vocab_size'],
                num_true=1,
                remove_accidental_hits=True,
                partition_strategy='div',
                name='bw_loss'
            ))
            self.raw_loss = 0.5 * tf.add(self.fw_loss, self.bw_loss, name='train_loss')
            if self.alpha > 0.0:
                self.activate_reg = tf.multiply(
                    self.alpha,
                    tf.add(
                        tf.div(
                            tf.reduce_sum(tf.square(self.model_train.fw_model['rnn_outputs'])),
                            tf.multiply(
                                tf.reduce_sum(self.model_train.seq_masks),
                                tf.to_float(self.model_train.fw_model['rnn_outputs'].shape[-1])
                            )
                        ),
                        tf.div(
                            tf.reduce_sum(tf.square(self.model_train.bw_model['rnn_outputs'])),
                            tf.multiply(
                                tf.reduce_sum(self.model_train.seq_masks),
                                tf.to_float(self.model_train.bw_model['rnn_outputs'].shape[-1])
                            )
                        )
                    )
                )
            else:
                self.activate_reg = None
            if self.beta > 0.0:
                self.temporal_activate_reg = tf.multiply(
                    self.beta,
                    tf.add(
                        tf.div(
                            tf.reduce_sum(tf.square(
                                tf.subtract(
                                    self.model_train.fw_model['rnn_outputs'][1:],
                                    self.model_train.fw_model['rnn_outputs'][:-1]
                                )
                            )),
                            tf.multiply(
                                tf.reduce_sum(self.model_train.seq_masks[1:]),
                                tf.to_float(self.model_train.fw_model['rnn_outputs'].shape[-1])
                            )
                        ),
                        tf.div(
                            tf.reduce_sum(tf.square(
                                tf.subtract(
                                    self.model_train.bw_model['rnn_outputs'][1:],
                                    self.model_train.bw_model['rnn_outputs'][:-1]
                                )
                            )),
                            tf.multiply(
                                tf.reduce_sum(self.model_train.seq_masks[1:]),
                                tf.to_float(self.model_train.bw_model['rnn_outputs'].shape[-1])
                            )
                        )
                    )
                )
            else:
                self.temporal_activate_reg = None
            if self.wdecay > 0.0:
                self.l2_reg = self.wdecay * tf.add_n([tf.reduce_sum(tf.square(x)) for x in self.model_train.variables], name='l2_reg')
            else:
                self.l2_reg = None
            self.loss = tf.add_n([x for x in (self.raw_loss, self.activate_reg, self.temporal_activate_reg, self.l2_reg) if x is not None], name='all_loss')
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = name2optimizer[self.optimizer['name']](**self.optimizer['params'], learning_rate=self.lr) if isinstance(self.optimizer, dict) else self.optimizer(self.lr)
            self.grads, self.vars = zip(*self.optimizer.compute_gradients(self.loss))
            if isinstance(self.clip_norm, float):
                self.grads, _ = tf.clip_by_global_norm(self.grads, clip_norm=self.clip_norm)
            self.grads = [tf.clip_by_value(grad, self.clip_min, self.clip_max) for grad in self.grads]  # Clip gradient between -1.0 and 1.0 just to be safe
            self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.vars), global_step=self.global_step)
            # Add summary op
            self.ppl = tf.exp(self.raw_loss)
            self.bpc = self.raw_loss/tf.log(2.0)
        train_summaries = [tf.summary.scalar('Loss', self.raw_loss),
                           tf.summary.scalar('forward_loss', self.fw_loss),
                           tf.summary.scalar('backward_loss', self.bw_loss),
                           tf.summary.scalar('Perplexity', self.ppl),
                           tf.summary.scalar('Bit_per_character', self.bpc),
                           tf.summary.scalar('Learning_rate', self.lr)]
        self.train_summaries = tf.summary.merge(train_summaries, name='train_summaries')
        if self.use_ema:
            ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay, num_updates=self.global_step)
            var_class = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_train.name)
            with tf.control_dependencies([self.train_op]):
                self.train_op = ema.apply(var_class)
            self.model_test = LanguageModel(**self.model_configs, reuse=True, is_training=False, custom_getter=get_getter(ema), name=self.model_train.name, is_cpu=False)
            self.test_saver = tf.train.Saver({v.op.name: ema.average(v) for v in var_class}, max_to_keep=100)
            self.ema = ema
        else:
            self.model_test = LanguageModel(**self.model_configs, reuse=True, is_training=False, custom_getter=None, name=self.model_train.name, is_cpu=False)
            self.test_saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        self.model_test.build_model()
        self.test_loss = 0.5 * tf.add(
            tf.contrib.seq2seq.sequence_loss(
                logits=self.model_test.fw_model['decoder'],
                targets=self.fw_y,
                weights=self.model_test.seq_masks,
                average_across_timesteps=True,
                average_across_batch=True),
            tf.contrib.seq2seq.sequence_loss(
                logits=self.model_test.bw_model['decoder'],
                targets=self.bw_y,
                weights=self.model_test.seq_masks,
                average_across_timesteps=True,
                average_across_batch=True),
            name='test_loss'
        )
        test_summaries = [tf.summary.scalar('Loss', self.test_loss),
                          tf.summary.scalar('Perplexity', tf.exp(self.test_loss)),
                          tf.summary.scalar('Bit_per_character', self.test_loss/tf.log(2.0))]
        self.test_summaries = tf.summary.merge(test_summaries, name='test_summaries')
        self.train_summaries_writer = tf.summary.FileWriter(
            self.train_summary_dir,
            self.session.graph
        )
        self.dev_summaries_writer = tf.summary.FileWriter(
            self.test_summary_dir,
            self.session.graph
        )
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.checkpoint_dir, folder_name))
        self.session.run(tf.global_variables_initializer())
        black_list_var = tf.get_collection(LSTM_SAVED_STATE)
        if not save_optimizer_var:
            black_list_var.extend(self.optimizer.variables())
        var2save = [x for x in tf.global_variables() if x not in black_list_var]
        self.train_saver = tf.train.Saver(var2save, max_to_keep=1)
        if latest_checkpoint is not None:
            self.train_saver.restore(self.session, latest_checkpoint)

    def train_step_classifier_tagger(self, train_char, train_labels, train_seq_labels, batch_size, lr, bptt, splits, folder_name, fine_tune_rate=None):
        start_time = time.time()
        save_path = os.path.join(self.checkpoint_dir, folder_name, 'model.cpkt')
        total_loss = 0
        for char_inputs, seq_lens, char_lens, true_labels, true_tags in get_batch_classifier_and_tagger(train_char, train_labels, train_seq_labels, batch_size, splits):
            real_bptt = get_random_bptt(bptt)
            fd = {
                self.lr: next(lr) if isinstance(lr, types.GeneratorType) else lr,
                self.model_train.inputs: char_inputs, self.model_train.seq_lens: seq_lens,
                self.model_train.char_lens: char_lens, self.model_train.bptt: real_bptt,
                self.true_y: true_labels, self.true_seq: true_tags
            }
            if self.fine_tune:
                fd.update(x for x in zip(self.fine_tune_rate, fine_tune_rate))
            _, train_loss, train_class_acc, train_tag_acc, step = self.session.run([self.train_op, self.loss, self.train_class_acc, self.train_tag_acc, self.global_step], feed_dict=fd)
            total_loss += train_loss * len(true_labels)
            self.logger.info("Step {:4d}: loss: {:05.5f}, class acc: {:05.5f}, tag acc: {:05.5f}, bptt: {:3d}, time {:05.2f}".format(
                step, train_loss, train_class_acc, train_tag_acc, real_bptt, time.time()-start_time))
            if step % self.save_freq == 0:
                self.train_saver.save(self.session, save_path, global_step=step)
        self.train_saver.save(self.session, save_path, global_step=step)
        total_loss /= len(train_labels)
        return True if total_loss <= 1e-4 else False

    def eval_step_classifier_tagger(self, test_char, test_labels, test_seq_labels, batch_size, bptt, splits, folder_name):
        start_time = time.time()
        save_path = os.path.join(self.checkpoint_dir, folder_name, 'model.cpkt')
        self.test_saver.save(self.session, save_path, global_step=self.session.run(self.global_step))
        total_loss = 0
        total_class_acc = 0
        total_tag_acc = 0
        count = 0
        tag_count = 0
        for char_inputs, seq_lens, char_lens, true_labels, true_tags in get_batch_classifier_and_tagger(test_char, test_labels, test_seq_labels, batch_size, splits, is_training=False):
            fd = {
                self.model_test.inputs: char_inputs, self.model_test.seq_lens: seq_lens,
                self.model_test.char_lens: char_lens, self.model_test.bptt: bptt,
                self.true_y: true_labels, self.true_seq: true_tags
            }
            test_loss, test_class_acc, test_tag_acc = self.session.run([self.test_loss, self.test_class_acc, self.test_tag_acc], feed_dict=fd)
            total_loss += test_loss * len(true_labels)
            total_class_acc += test_class_acc * len(true_labels)
            total_tag_acc += test_tag_acc * np.sum(seq_lens)
            count += len(true_labels)
            tag_count += np.sum(seq_lens)
            self.logger.info("Evaluate total loss: {:05.5f}, total class acc: {:05.5f}, total tag acc: {:05.5f}, time {:05.2f}".format(
                total_loss/count, total_class_acc/count, total_tag_acc/tag_count, time.time()-start_time))

    def train_step(self, model, train_word, train_char, lr, start_i=0, folder_name='train', fine_tune_rate=None):
        start_time = time.time()
        batch, i = 0, start_i
        step = None
        total_len = len(train_word)
        while i < total_len-1:
            (fw_x, fw_cl, fw_y), (bw_x, bw_cl, bw_y) = get_batch(train_word, train_char, bptt=self.bptt, i=i)
            fd = {
                self.lr: lr,
                model.fw_inputs: fw_x,
                model.bw_inputs: bw_x,
                model.fw_char_lens: fw_cl,
                model.bw_char_lens: bw_cl,
                self.fw_y: fw_y,
                self.bw_y: bw_y,
                model.seq_lens: [fw_y.shape[0]]*fw_y.shape[1],
                model.reset_state: i == start_i
            }
            if self.fine_tune:
                fd.update(x for x in zip(self.fine_tune_rate, fine_tune_rate))
            _, fwl, bwl, ppl, bpc, step, summaries = self.session.run(
                [self.train_op, self.fw_loss, self.bw_loss, self.ppl, self.bpc, self.global_step, self.train_summaries],
                feed_dict=fd
            )
            self.train_summaries_writer.add_summary(summaries, step)
            i += len(fw_y)
            self.logger.info(
                "Step {:4d}: progress {}/{},forward loss {:05.5f}, backward loss {:05.5f}, ppl {:05.2f}, bpc {:05.2f}, time {:05.2f}".format(
                    step, i, total_len,
                    fwl, bwl,
                    ppl,
                    bpc,
                    time.time()-start_time)
            )
            # start_time = time.time()
            batch += 1
            if step % self.save_freq == 0:
                self.train_saver.save(self.session, os.path.join(self.checkpoint_dir, folder_name, 'model.cpkt'), global_step=step)
        self.train_saver.save(self.session, os.path.join(self.checkpoint_dir, folder_name, 'model.cpkt'), global_step=step)

    def evaluate_step(self, model, test_word, test_char, folder_name='test'):
        start_time = time.time()
        total_loss = 0
        step = None
        self.test_saver.save(self.session, os.path.join(self.checkpoint_dir, folder_name, 'model.cpkt'), global_step=self.session.run(self.global_step))
        self.train_saver.save(self.session, os.path.join(self.checkpoint_dir, folder_name, 'model-full.cpkt'), global_step=self.session.run(self.global_step))
        for i in range(0, len(test_word), self.bptt):
            (fw_x, fw_cl, fw_y), (bw_x, bw_cl, bw_y) = get_batch(test_word, test_char, self.bptt, i, evaluate=True)
            summaries, loss, step = self.session.run(
                [self.test_summaries, self.test_loss, self.global_step],
                feed_dict={
                    model.fw_inputs: fw_x, model.fw_char_lens: fw_cl,
                    model.bw_inputs: bw_x, model.bw_char_lens: bw_cl,
                    self.fw_y: fw_y,
                    self.bw_y: bw_y,
                    model.seq_lens: [fw_y.shape[0]]*fw_y.shape[1],
                    model.reset_state: i == 0
                }
            )
            self.dev_summaries_writer.add_summary(summaries, step)
            total_loss += loss * len(fw_y)
            self.logger.info("Evaluate loss {}, time {}".format(loss, time.time()-start_time))
        total_loss /= len(test_word)
        self.logger.info("Evaluate total loss {}, time {}".format(total_loss, time.time()-start_time))

    def train_dev_loop(self, train_word, train_char, test_word, test_char, lr, fine_tune_rate=None, start_i=0, folder_train='train', folder_test='test'):
        self.train_step(self.model_train, train_word, train_char, lr, start_i=start_i, fine_tune_rate=fine_tune_rate, folder_name=folder_train)
        self.evaluate_step(self.model_test, test_word, test_char, folder_name=folder_test)

    def close(self):
        self.session.close()
        [x.close() for x in self.logger.handlers]
