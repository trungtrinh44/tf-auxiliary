import tensorflow as tf
from utils import get_batch, get_getter, get_logger
import time
from model import LanguageModel


class Trainer():
    def __init__(self, model_configs, optimizer, learning_rate, decay_rate, decay_freq, alpha, beta, clip_norm, bptt, log_path, train_summary_dir, test_summary_dir, checkpoint_dir, name='LM_Trainer'):
        self.model_configs = model_configs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
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

    def build(self):
        self.session = tf.Session()
        self.model_train = LanguageModel(
            **self.model_configs, reuse=False, is_training=True)
        self.model_train.build_model()
        with tf.variable_scope(self.name):
            self.y = tf.placeholder(dtype=tf.int32, shape=[
                                    None, None], name='y')
            self.raw_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.model_train.decoder,
                targets=self.y,
                weights=self.model_train.seq_masks,
                average_across_timesteps=True,
                average_across_batch=True,
                name='train_loss'
            )  # Since we try the character model first, simple loss is the best
            self.activate_reg = tf.multiply(
                self.alpha,
                tf.div(
                    tf.reduce_sum(tf.square(self.model_train.rnn_outputs)),
                    tf.multiply(
                        tf.reduce_sum(self.model_train.expand_seq_masks),
                        tf.to_float(self.model_train.rnn_outputs.shape[-1])
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
                        tf.to_float(self.model_train.rnn_outputs.shape[-1])
                    )
                )
            )
            self.loss = tf.add_n(
                [self.raw_loss, self.activate_reg, self.temporal_activate_reg], name='all_loss')
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                            self.decay_freq, self.decay_rate, staircase=False)
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
            self.bpc = self.raw_loss/tf.log(2.0)
        train_summaries = [tf.summary.scalar('Loss', self.raw_loss),
                           tf.summary.scalar('Perplexity', self.ppl),
                           tf.summary.scalar('Bit_per_character', self.bpc)]
        self.train_summaries = tf.summary.merge(
            train_summaries, name='train_summaries')
        ema = tf.train.ExponentialMovingAverage(decay=0.998)
        var_class = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.model_train.name)
        with tf.control_dependencies([self.train_op]):
            self.train_op = ema.apply(var_class)
        self.model_test = LanguageModel(
            **self.model_configs, reuse=True, is_training=False, custom_getter=get_getter(ema), name=self.model_train.name)
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
        self.train_saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=1
        )
        self.test_saver = tf.train.Saver(
            {v.op.name: ema.average(v) for v in var_class}, max_to_keep=1000
        )
        self.train_summaries_writer = tf.summary.FileWriter(
            self.train_summary_dir,
            self.session.graph
        )
        self.dev_summaries_writer = tf.summary.FileWriter(
            self.test_summary_dir,
            self.session.graph
        )

    def train_step(self, train_data):
        start_time = time.time()
        batch, i = 0, 0
        step = None
        while i < len(train_data)-2:
            next_x, next_y = get_batch(train_data, self.bptt, i)
            _, loss, ppl, bpc, step, summaries = self.session.run(
                [self.train_op, self.raw_loss, self.ppl, self.bpc,
                    self.global_step, self.train_summaries],
                feed_dict={
                    self.model_train.inputs: next_x,
                    self.y: next_y,
                    self.model_train.seq_lens: [
                        next_x.shape[0]]*next_x.shape[1]
                }
            )
            self.train_summaries_writer.add_summary(summaries, step)
            self.logger.info(
                "Step {}: loss {}, ppl {}, bpc {}, time {}".format(step, loss, ppl, bpc, time.time()-start_time))
            start_time = time.time()

            batch += 1
            i += len(next_y)
        self.train_saver.save(
            self.session, self.checkpoint_dir+'/train', global_step=step)

    def evaluate_step(self, test_data):
        start_time = time.time()
        total_loss = 0
        step = None
        for i in range(0, len(test_data), self.bptt):
            next_x, next_y = get_batch(test_data, self.bptt, i, evaluate=True)
            summaries, loss, step = self.session.run(
                [self.test_summaries, self.test_loss, self.global_step],
                feed_dict={
                    self.model_test.inputs: next_x,
                    self.y: next_y,
                    self.model_test.seq_lens: [
                        next_x.shape[0]]*next_x.shape[1]
                }
            )
            self.test_summaries(summaries, step)
            total_loss += loss
        self.logger.info("Evaluate loss {}, time {}".format(
            loss, time.time()-start_time))
        self.test_saver.save(
            self.session, self.checkpoint_dir+'/test', global_step=step)

    def train_dev_loop(self, train_data, test_data):
        self.train_step(train_data)
        self.evaluate_step(test_data)

    def close(self):
        self.session.close()


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
