import tensorflow as tf
from utils import get_batch, get_getter
import time
from model import LanguageModel


class Trainer():
    def __init__(self, model_configs, optimizer, learning_rate, y, alpha, beta, clip_norm, bptt, train_summary_dir, test_summary_dir, checkpoint_dir, name='LM_Trainer'):
        self.model_configs = model_configs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.y = y
        self.name = name
        self.train_summary_dir = train_summary_dir
        self.test_summary_dir = test_summary_dir
        self.checkpoint_dir = checkpoint_dir
        self.alpha = alpha
        self.beta = beta
        self.bptt = bptt
        self.clip_norm = clip_norm

    def build(self):
        self.session = tf.Session()
        self.model_train = LanguageModel(
            **self.model_configs, reuse=False, is_training=True)
        with tf.variable_scope(self.name):
            self.raw_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.model_train.decoder,
                targets=self.y,
                weights=self.model_train.seq_masks,
                average_across_timesteps=True,
                average_across_batch=True
            )  # Since we try the character model first, simple loss is the best
            self.activate_reg = tf.multiply(
                self.alpha,
                tf.div(
                    tf.reduce_sum(tf.square(self.model_train.rnn_outputs)),
                    tf.multiply(
                        tf.reduce_sum(self.model_train.expand_seq_masks),
                        self.model_train.rnn_outputs.shape[-1]
                    )
                )
            )
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
                        tf.reduce_sum(self.model_train.expand_seq_masks[1:]),
                        self.model_train.rnn_outputs.shape[-1]
                    )
                )
            )
            self.loss = self.raw_loss + self.activate_reg + self.temporal_activate_reg
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            self.optimizer = self.optimizer(self.learning_rate)
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
            self.bpc = self.raw_loss/tf.log(2)
            tf.summary.scalar('Loss', self.raw_loss)
            tf.summary.scalar('Perplexity', self.ppl)
            tf.summary.scalar('Bit per character', self.bpc)
            self.summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.train_summary_dir,
                                                      self.session.graph)
            self.test_writer = tf.summary.FileWriter(self.test_summary_dir)
        ema = tf.train.ExponentialMovingAverage(decay=0.998)
        var_class = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.model_train.name)
        with tf.control_dependencies([self.train_op]):
            self.train_op = ema.apply(var_class)
        self.model_test = LanguageModel(
            **self.model_configs, reuse=True, is_training=False, custom_getter=get_getter(ema), name=self.model_train.name)
        self.train_saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=1000
        )
        self.test_saver = tf.train.Saver(
            {v.op.name: ema.average(v) for v in var_class}, max_to_keep=1000
        )

    def train_step(self, train_data):
        total_loss = 0
        start_time = time.time()
        batch, i = 0, 0
        while i < len(train_data)-2:
            next_x, next_y = get_batch(train_data, self.bptt, i)

            batch += 1
            i += len(next_y)

    def close(self):
        self.session.close()
