# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import os
import time

import tensorflow as tf

from classifier import Classifier
from model import LanguageModel
from utils import get_batch, get_getter, get_logger, optimistic_restore


class Trainer():
    def __init__(self, model_configs,
                 optimizer,
                 decay_rate,
                 decay_freq,
                 wdecay,
                 alpha,
                 beta,
                 clip_norm,
                 bptt,
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
        self.logger = get_logger(log_path)
        self.decay_rate = decay_rate
        self.decay_freq = decay_freq
        self.use_ema = use_ema
        self.save_freq = save_freq
        self.wdecay = wdecay
        self.ema_decay = ema_decay

    def initialize_uninitialized(self):
        global_vars = tf.global_variables()
        is_not_initialized = self.session.run(
            [tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(
            global_vars, is_not_initialized) if not f]

        # only for testing
        self.logger.info(str([str(i.name) for i in not_initialized_vars]))
        if len(not_initialized_vars):
            self.session.run(tf.variables_initializer(not_initialized_vars))

    def build(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        self.session = tf.Session(config=config)
        self.model_train = LanguageModel(
            **self.model_configs, reuse=False, is_training=True)
        self.model_train.build_model()
        self.logger.info(self.model_train.variables)
        with tf.variable_scope(self.name):
            self.y = tf.placeholder(dtype=tf.int32, shape=[
                                    None, None], name='y')
            self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
            self.raw_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.model_train.decoder,
                targets=self.y,
                weights=self.model_train.seq_masks,
                average_across_timesteps=True,
                average_across_batch=True,
                name='train_loss'
            )  # Since we try the character model first, simple loss is the best
            if self.alpha > 0.0:
                self.activate_reg = tf.multiply(
                    self.alpha,
                    tf.div(
                        tf.reduce_sum(tf.square(self.model_train.rnn_outputs)),
                        tf.multiply(
                            tf.reduce_sum(self.model_train.seq_masks),
                            tf.to_float(self.model_train.rnn_outputs.shape[-1])
                        )
                    )
                )
            else:
                self.activate_reg = None
            if self.beta > 0.0:
                self.temporal_activate_reg = tf.multiply(
                    self.beta,
                    tf.div(
                        tf.reduce_sum(tf.square(
                            tf.subtract(
                                self.model_train.rnn_outputs[1:],
                                self.model_train.rnn_outputs[:-1]
                            )
                        )),
                        tf.multiply(
                            tf.reduce_sum(self.model_train.seq_masks[1:]),
                            tf.to_float(self.model_train.rnn_outputs.shape[-1])
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
                           tf.summary.scalar('Perplexity', self.ppl),
                           tf.summary.scalar('Bit_per_character', self.bpc),
                           tf.summary.scalar('Learning_rate', self.learning_rate)]
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
                {v.op.name: ema.average(v) for v in var_class}, max_to_keep=1000
            )
            self.ema = ema
        else:
            self.model_test = LanguageModel(
                **self.model_configs, reuse=True, is_training=False, custom_getter=None, name=self.model_train.name)
            self.test_saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=1000
            )
        self.model_test.build_model()
        self.test_loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.model_test.decoder,
            targets=self.y,
            weights=self.model_test.seq_masks,
            average_across_timesteps=True,
            average_across_batch=True,
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
        os.makedirs(os.path.join(self.checkpoint_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, 'test'), exist_ok=True)
        latest_checkpoint = tf.train.latest_checkpoint(
            os.path.join(self.checkpoint_dir, 'train'))
        self.session.run(tf.global_variables_initializer())
        lstm_saved_state = tf.get_collection('LSTM_SAVED_STATE')
        self.train_saver = tf.train.Saver(
            [x for x in tf.global_variables() if x not in lstm_saved_state], max_to_keep=1
        )
        if latest_checkpoint is not None:
            self.train_saver.restore(self.session, latest_checkpoint)

    def train_step(self, model, train_data, lr, folder_name='train'):
        start_time = time.time()
        batch, i = 0, 0
        step = None
        while i < len(train_data)-1:
            next_x, next_y = get_batch(train_data, self.bptt, i)
            self.logger.info("Len {:4d}".format(len(next_x)))
            _, loss, ppl, bpc, step, summaries = self.session.run(
                [self.train_op, self.raw_loss, self.ppl, self.bpc,
                    self.global_step, self.train_summaries],
                feed_dict={
                    self.lr: lr,
                    model.inputs: next_x,
                    self.y: next_y,
                    model.seq_lens: [
                        next_x.shape[0]]*next_x.shape[1],
                    model.reset_state: i == 0
                }
            )
            self.train_summaries_writer.add_summary(summaries, step)
            self.logger.info(
                "Step {:4d}: loss {:05.5f}, ppl {:05.2f}, bpc {:05.2f}, time {:05.2f}".format(
                    step,
                    loss,
                    ppl,
                    bpc,
                    time.time()-start_time)
            )
            start_time = time.time()
            batch += 1
            i += len(next_y)
            if step % self.save_freq == 0:
                self.train_saver.save(
                    self.session, os.path.join(self.checkpoint_dir, folder_name, 'model.cpkt'), global_step=step)
        self.train_saver.save(
            self.session, os.path.join(self.checkpoint_dir, 'train', 'model.cpkt'), global_step=step)

    def evaluate_step(self, model, test_data, folder_name='test'):
        start_time = time.time()
        total_loss = 0
        step = None
        for i in range(0, len(test_data), self.bptt):
            next_x, next_y = get_batch(test_data, self.bptt, i, evaluate=True)
            summaries, loss, step = self.session.run(
                [self.test_summaries, self.test_loss, self.global_step],
                feed_dict={
                    model.inputs: next_x,
                    self.y: next_y,
                    model.seq_lens: [
                        next_x.shape[0]]*next_x.shape[1],
                    model.reset_state: i == 0
                }
            )
            self.dev_summaries_writer.add_summary(summaries, step)
            total_loss += loss * len(next_x)
            self.logger.info("Evaluate loss {}, time {}".format(
                loss, time.time()-start_time))
        total_loss /= len(test_data)
        self.logger.info("Evaluate total loss {}, time {}".format(
            total_loss, time.time()-start_time))
        self.test_saver.save(
            self.session, os.path.join(self.checkpoint_dir, folder_name, 'model.cpkt'), global_step=step)

    def train_dev_loop(self, train_data, test_data, lr):
        self.train_step(self.model_train, train_data, lr)
        self.evaluate_step(self.model_test, test_data)

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

    def fine_tune_train_dev_loop(self, train_data, test_data, lr):
        assert self.fine_tune_model, 'Make sure that you have run build_dicriminative_fine_tuning_lm_model.'
        self.train_step(self.fine_tune_model, train_data, lr,
                        folder_name='fine_tune_train')
        self.evaluate_step(self.fine_tune_model, test_data,
                           folder_name='fine_tune_test')

    def add_classifier(self, num_classes, classifier_configs):
        self.num_classes = num_classes
        self.logger.info('Classifier configs: {}'.format(classifier_configs))
        # TODO: Add discriminative fine tuning
        inputs = tf.stop_gradient(self.model_test.rnn_outputs)
        inputs = tf.transpose(inputs, [1, 0, 2])
        self.train_classifiers = [
            Classifier(**classifier_configs, inputs=inputs,
                       num_class=2, is_training=True, reuse=False, name='Classifier_' + str(i)) for i in range(num_classes)
        ]
        for c in self.train_classifiers:
            c.build()
        self.class_ys = [
            tf.placeholder(dtype=tf.float32, shape=[None, 2], name='class_y'+str(i)) for i in range(num_classes)
        ]
        self.train_class_losses = [
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=c.outputs)) for y_true, c in zip(self.class_ys, self.train_classifiers)
        ]
        self.all_train_class_loss = tf.add_n(
            self.train_class_losses, name='all_train_class_loss')
        self.test_classifiers = [
            Classifier(**classifier_configs, inputs=inputs,
                       num_class=2, is_training=False, reuse=True, name='Classifier_' + str(i)) for i in range(num_classes)
        ]
        reg_loss = [x for c in self.train_classifiers for x in c.l2_reg_losses]
        if len(reg_loss) > 0:
            self.class_real_loss = tf.add(
                self.all_train_class_loss,
                tf.add_n(reg_loss),
                name='class_real_loss'
            )
        else:
            self.class_real_loss = tf.identity(
                self.all_train_class_loss, name='class_real_loss')
        for c in self.test_classifiers:
            c.build()
        self.test_class_losses = [
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=c.outputs)) for y_true, c in zip(self.class_ys, self.test_classifiers)
        ]
        self.all_test_class_loss = tf.add_n(
            self.test_class_losses, name='all_test_class_loss')
        self.test_class_saver = tf.train.Saver(
            dict(**{v.op.name: v for c in self.test_classifiers for v in c.variables},
                 **self.test_saver._var_list),
            max_to_keep=1000
        )
        self.train_class_saver = tf.train.Saver(
            tf.global_variables(),
            max_to_keep=1
        )
        self.class_global_step = tf.Variable(
            0, name="class_global_step", trainable=False)
        self.class_grads, self.class_vars = zip(
            *self.optimizer.compute_gradients(self.class_real_loss))
        self.class_grads, _ = tf.clip_by_global_norm(
            self.class_grads, clip_norm=self.clip_norm)
        self.class_train_op = self.optimizer.apply_gradients(
            zip(self.class_grads, self.class_vars),
            global_step=self.class_global_step
        )
        self.initialize_uninitialized()

    def classifier_train_dev_loop(self, train_gen, test_gen):
        self.classifier_train_step(train_gen)
        self.classifier_evaluate_step(test_gen)

    def classifier_train_step(self, train_gen):
        start_time = time.time()
        step = None
        for next_x, next_y, seq_len in train_gen:
            fd = {
                self.model_test.inputs: next_x,
                self.model_test.seq_lens: seq_len,
                self.model_test.reset_state: True
            }
            fd.update(
                (k, v) for k, v in zip(self.class_ys, next_y)
            )
            _, loss, step = self.session.run(
                [self.class_train_op, self.all_train_class_loss,
                    self.class_global_step],
                feed_dict=fd)
            self.logger.info(
                "Step {:4d}: loss {:05.5f}, time {:05.2f}".format(
                    step,
                    loss,
                    time.time()-start_time)
            )
            start_time = time.time()
            if step % self.save_freq == 0:
                self.train_class_saver.save(
                    self.session, os.path.join(self.checkpoint_dir, 'classifier', 'train', 'model.cpkt'), global_step=step)
        self.train_class_saver.save(
            self.session, os.path.join(self.checkpoint_dir, 'classifier', 'train', 'model.cpkt'), global_step=step)

    def classifier_evaluate_step(self, test_gen):
        start_time = time.time()
        step = None
        total_loss = 0.0
        batch = 0
        for i, (next_x, next_y, seq_len) in enumerate(test_gen):
            fd = {
                self.model_test.inputs: next_x,
                self.model_test.seq_lens: seq_len,
                self.model_test.reset_state: True
            }
            fd.update(
                (k, v) for k, v in zip(self.class_ys, next_y)
            )
            loss, step = self.session.run(
                [self.all_test_class_loss,
                    self.class_global_step],
                feed_dict=fd)
            self.logger.info(
                "Evaluate step {:4d}: loss {:05.5f}, time {:05.2f}".format(
                    step,
                    loss,
                    time.time()-start_time)
            )
            total_loss += len(next_x) * loss
            batch += len(next_x)
        self.test_class_saver.save(
            self.session, os.path.join(self.checkpoint_dir, 'classifier', 'test', 'model.cpkt'), global_step=step)
        self.logger.info(
            'Evaluate total loss: {}, avg. time: {}'.format(total_loss / batch,
                                                            (time.time()-start_time) / i)
        )


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
