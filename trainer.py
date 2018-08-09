import tensorflow as tf


class Trainer():
    def __init__(self, model, optimizer, learning_rate, y, train_summary_dir, test_summary_dir, name='LM_Trainer'):
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.y = y
        self.name = name
        self.train_summary_dir = train_summary_dir
        self.test_summary_dir = test_summary_dir

    def build(self):
        self.session = tf.Session()
        with tf.variable_scope(self.name):
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.model.decoder,
                targets=self.y,
                weights=self.model.seq_masks,
                average_across_timesteps=True,
                average_across_batch=True
            )
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            self.optimizer = self.optimizer(self.learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars,
                global_step=self.global_step
            )
            # Add summary op
            self.ppl = tf.exp(self.loss)
            self.bpc = self.loss/tf.log(2)
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Perplexity', self.ppl)
            tf.summary.scalar('Bit per character', self.bpc)
            self.summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.train_summary_dir,
                                                      self.session.graph)
            self.test_writer = tf.summary.FileWriter(self.test_summary_dir)

    def close(self):
        self.session.close()
