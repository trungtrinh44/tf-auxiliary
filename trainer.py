# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import os
import time
import json

import tensorflow as tf

from classifier import Classifier
from model_v2 import LSTM_SAVED_STATE, LanguageModel
from utils import get_batch, get_getter, get_logger


class Trainer():
    def __init__(self, model_configs,
                 optimizer,
                 wdecay,
                 alpha,
                 beta,
                 clip_norm,
                 bptt,
                 negative_samples,
                 log_path,
                 train_summary_dir,
                 test_summary_dir,
                 checkpoint_dir,
                 save_freq,
                 use_ema=False,
                 ema_decay=0.998,
                 name='LM_Trainer'):
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
        os.makedirs(os.path.join(self.checkpoint_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, 'test'), exist_ok=True)
        self.logger = get_logger(log_path)
        self.use_ema = use_ema
        self.save_freq = save_freq
        self.wdecay = wdecay
        self.ema_decay = ema_decay
        self.negative_samples = negative_samples

    def build(self):
        with open(os.path.join(self.checkpoint_dir, 'model_configs.json'), 'w') as out:
            json.dump(self.model_configs, out)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        self.session = tf.Session(config=config)
        self.model_train = LanguageModel(
            **self.model_configs, reuse=False, is_training=True)
        self.model_train.build_model()
        with tf.variable_scope(self.name):
            self.fw_y = tf.placeholder(dtype=tf.int32, shape=[
                None, None], name='fw_y')
            self.bw_y = tf.placeholder(dtype=tf.int32, shape=[
                None, None], name='bw_y')
            self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
            self.fw_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=tf.transpose(
                    self.model_train.share_decode_W, (1, 0)),
                biases=self.model_train.share_decode_b,
                labels=tf.reshape(self.fw_y, [-1, 1]),
                inputs=tf.reshape(
                    self.model_train.fw_model['rnn_outputs'], (-1, self.model_train.fw_model['rnn_outputs'].shape[-1])),
                num_sampled=self.negative_samples,
                num_classes=self.model_configs['vocab_size'],
                num_true=1,
                remove_accidental_hits=True,
                partition_strategy='div',
                name='fw_loss'
            ))
            self.bw_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=tf.transpose(
                    self.model_train.share_decode_W, (1, 0)),
                biases=self.model_train.share_decode_b,
                labels=tf.reshape(self.bw_y, [-1, 1]),
                inputs=tf.reshape(
                    self.model_train.bw_model['rnn_outputs'], (-1, self.model_train.bw_model['rnn_outputs'].shape[-1])),
                num_sampled=self.negative_samples,
                num_classes=self.model_configs['vocab_size'],
                num_true=1,
                remove_accidental_hits=True,
                partition_strategy='div',
                name='bw_loss'
            ))
            self.raw_loss = 0.5 * tf.add(self.fw_loss, self.bw_loss, name='train_loss')
#             self.raw_loss = tf.reduce_mean(self.raw_loss)
            if self.alpha > 0.0:
                self.activate_reg = tf.multiply(
                    self.alpha,
                    tf.add(
                        tf.div(
                            tf.reduce_sum(
                                tf.square(self.model_train.fw_model['rnn_outputs'])),
                            tf.multiply(
                                tf.reduce_sum(self.model_train.seq_masks),
                                tf.to_float(
                                    self.model_train.fw_model['rnn_outputs'].shape[-1])
                            )
                        ),
                        tf.div(
                            tf.reduce_sum(
                                tf.square(self.model_train.bw_model['rnn_outputs'])),
                            tf.multiply(
                                tf.reduce_sum(self.model_train.seq_masks),
                                tf.to_float(
                                    self.model_train.bw_model['rnn_outputs'].shape[-1])
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
                                tf.to_float(
                                    self.model_train.fw_model['rnn_outputs'].shape[-1])
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
                                tf.to_float(
                                    self.model_train.bw_model['rnn_outputs'].shape[-1])
                            )
                        )
                    )
                )
            else:
                self.temporal_activate_reg = None
            if self.wdecay > 0.0:
                self.l2_reg = self.wdecay * \
                    tf.add_n([tf.reduce_sum(tf.square(x))
                              for x in self.model_train.variables], name='l2_reg')
            else:
                self.l2_reg = None
            self.loss = tf.add_n(
                [x for x in (self.raw_loss, self.activate_reg, self.temporal_activate_reg, self.l2_reg) if x is not None], name='all_loss')
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            self.optimizer = self.optimizer(self.lr)
            self.grads, self.vars = zip(
                *self.optimizer.compute_gradients(self.loss))
            self.grads, _ = tf.clip_by_global_norm(
                self.grads, clip_norm=self.clip_norm)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.grads, self.vars),
                global_step=self.global_step
            )
            # Add summary op
            self.ppl = tf.exp(self.raw_loss)
            self.bpc = self.raw_loss/tf.log(2.0)
        train_summaries = [tf.summary.scalar('Loss', self.raw_loss),
                           tf.summary.scalar('forward_loss', self.fw_loss),
                           tf.summary.scalar('backward_loss', self.bw_loss),
                           tf.summary.scalar('Perplexity', self.ppl),
                           tf.summary.scalar('Bit_per_character', self.bpc),
                           tf.summary.scalar('Learning_rate', self.lr)]
        self.train_summaries = tf.summary.merge(
            train_summaries, name='train_summaries')
        if self.use_ema:
            ema = tf.train.ExponentialMovingAverage(
                decay=self.ema_decay, num_updates=self.global_step)
            var_class = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.model_train.name)
            with tf.control_dependencies([self.train_op]):
                self.train_op = ema.apply(var_class)
            self.model_test = LanguageModel(
                **self.model_configs, reuse=True, is_training=False, custom_getter=get_getter(ema), name=self.model_train.name)
            self.test_saver = tf.train.Saver(
                {v.op.name: ema.average(v) for v in var_class}, max_to_keep=100
            )
            self.ema = ema
        else:
            self.model_test = LanguageModel(
                **self.model_configs, reuse=True, is_training=False, custom_getter=None, name=self.model_train.name)
            self.test_saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=100
            )
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
                          tf.summary.scalar(
            'Perplexity', tf.exp(self.test_loss)),
            tf.summary.scalar('Bit_per_character', self.test_loss/tf.log(2.0))]
        self.test_summaries = tf.summary.merge(
            test_summaries, name='test_summaries')
        self.train_summaries_writer = tf.summary.FileWriter(
            self.train_summary_dir,
            self.session.graph
        )
        self.dev_summaries_writer = tf.summary.FileWriter(
            self.test_summary_dir,
            self.session.graph
        )
        latest_checkpoint = tf.train.latest_checkpoint(
            os.path.join(self.checkpoint_dir, 'train'))
        self.session.run(tf.global_variables_initializer())
        lstm_saved_state = tf.get_collection(LSTM_SAVED_STATE)
        self.train_saver = tf.train.Saver(
            [x for x in tf.global_variables() if x not in lstm_saved_state], max_to_keep=1
        )
        if latest_checkpoint is not None:
            self.train_saver.restore(self.session, latest_checkpoint)

    def train_step(self, model, train_word, train_char, lr, start_i=0, folder_name='train'):
        start_time = time.time()
        batch, i = 0, start_i
        step = None
        total_len = len(train_word)
        train_word = train_word[i:]
        while i < total_len-1:
            (fw_x, fw_y), (bw_x, bw_y) = get_batch(
                train_word, train_char, bptt=self.bptt, i=i)
#             self.logger.info("Len {:4d}".format(len(fw_x)))
            _, fwl, bwl, ppl, bpc, step, summaries = self.session.run(
                [self.train_op, self.fw_loss, self.bw_loss, self.ppl, self.bpc,
                    self.global_step, self.train_summaries],
                feed_dict={
                    self.lr: lr,
                    model.fw_inputs: fw_x,
                    model.bw_inputs: bw_x,
                    self.fw_y: fw_y,
                    self.bw_y: bw_y,
                    model.seq_lens: [
                        fw_y.shape[0]]*fw_y.shape[1],
                    model.reset_state: i == 0
                }
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
            start_time = time.time()
            batch += 1
            if step % self.save_freq == 0:
                self.train_saver.save(self.session, os.path.join(self.checkpoint_dir, folder_name, 'model.cpkt'), global_step=step)
        self.train_saver.save(self.session, os.path.join(self.checkpoint_dir, 'train', 'model.cpkt'), global_step=step)

    def evaluate_step(self, model, test_word, test_char, folder_name='test'):
        start_time = time.time()
        total_loss = 0
        step = None
        self.test_saver.save(self.session, os.path.join(self.checkpoint_dir, folder_name, 'model.cpkt'), 
                             global_step=self.session.run(self.global_step))
        for i in range(0, len(test_word), self.bptt):
            (fw_x, fw_y), (bw_x, bw_y) = get_batch(
                test_word, test_char, self.bptt, i, evaluate=True)
            summaries, loss, step = self.session.run(
                [self.test_summaries, self.test_loss, self.global_step],
                feed_dict={
                    model.fw_inputs: fw_x,
                    model.bw_inputs: bw_x,
                    self.fw_y: fw_y,
                    self.bw_y: bw_y,
                    model.seq_lens: [
                        fw_y.shape[0]]*fw_y.shape[1],
                    model.reset_state: i == 0
                }
            )
            self.dev_summaries_writer.add_summary(summaries, step)
            total_loss += loss * len(fw_y)
            self.logger.info("Evaluate loss {}, time {}".format(
                loss, time.time()-start_time))
        total_loss /= len(test_word)
        self.logger.info("Evaluate total loss {}, time {}".format(
            total_loss, time.time()-start_time))

    def train_dev_loop(self, train_word, train_char, test_word, test_char, lr, start_i=0):
        self.train_step(self.model_train, train_word, train_char, lr, start_i=start_i)
        self.evaluate_step(self.model_test, test_word, test_char)

    def close(self):
        self.session.close()

    def build_dicriminative_fine_tuning_lm_model(self, fine_tune_lr):
        assert isinstance(fine_tune_lr, list), 'fine_tune_lr must be a list'
        self.fine_tune_lr = fine_tune_lr
        # Fine tune directly on EMA-model
        self.fine_tune_model = LanguageModel(
            **self.model_configs,
            reuse=True,
            is_training=True,
            custom_getter=get_getter(self.ema) if self.use_ema else None,
            fine_tune_lr=fine_tune_lr,
            name=self.model_train.name)

    def fine_tune_train_dev_loop(self, train_word, train_char, test_word, test_char, lr):
        assert self.fine_tune_model, 'Make sure that you have run build_dicriminative_fine_tuning_lm_model.'
        self.train_step(self.fine_tune_model, train_word, train_char,  lr,
                        folder_name='fine_tune_train')
        self.evaluate_step(self.fine_tune_model, test_word, test_char,
                           folder_name='fine_tune_test')


if __name__ == '__main__':
    params = dict(
        model_configs={
            'rnn_layers': [
                {'units': 1150, 'input_size': 400, 'drop_i': 0.5, 'drop_w': 0.4},
                {'units': 1150, 'input_size': 1150, 'drop_w': 0.4},
                {'units': 400, 'input_size': 1150, 'drop_o': 0.3, 'drop_w': 0.4}
            ],
            'vocab_size': 200,
            'drop_e': 0.1
        },
        optimizer=tf.train.GradientDescentOptimizer,
        learning_rate=30.0,
        decay_rate=0.1,
        decay_freq=100000,
        alpha=1e-5,
        beta=1e-5,
        clip_norm=1.0,
        bptt=200,
        log_path='logs/',
        train_summary_dir='train_summary/',
        test_summary_dir='test_summary/',
        checkpoint_dir='checkpoints/'
    )
    my_trainer = Trainer(**params)
    my_trainer.build()
