import tensorflow as tf


class Trainer():
    def __init__(self, model, optimizer, learning_rate, y, name='LM_Trainer'):
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.y = y
        self.name = name

    def build(self):
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
