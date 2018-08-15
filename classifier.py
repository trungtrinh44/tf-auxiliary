import tensorflow as tf


class Classifier():
    def __init__(self, inputs, hiddens, k, num_class, is_training, reuse, name='Classifier'):
        self.inputs = inputs
        self.hiddens = hiddens
        self.k = k
        self.num_class = num_class
        self.name = name
        self.is_training = is_training
        self.reuse = reuse

    def build(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            inputs = self.inputs
            bsz = tf.shape(inputs)[0]
            for i, layer in enumerate(self.hiddens):
                drop_i = layer.get('drop_i', 0.0)
                if drop_i > 0.0 and self.is_training:
                    inputs = tf.layers.dropout(inputs, drop_i,
                                               noise_shape=[
                                                   bsz, 1, inputs.shape[-1]],
                                               training=self.is_training)
                outputs = tf.layers.conv1d(
                    inputs=inputs,
                    filters=layer['filters'],
                    kernel_size=layer['kernel_size'],
                    strides=layer.get('strides', 1),
                    padding=layer.get('padding', 'valid'),
                    data_format='channels_last',
                    dilation_rate=layer.get('dilation_rate', 1),
                    activation=None,
                    use_bias=True,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                        layer['l2_reg']) if 'l2_reg' in layer and self.is_training else None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    trainable=True,
                    name='conv1d_' + str(i),
                )
                outputs = tf.layers.batch_normalization(
                    inputs=outputs,
                    axis=-1,
                    training=self.is_training,
                    fused=True
                )
                outputs = tf.nn.relu(outputs)
                drop_o = layer.get('drop_o', 0.0)
                if drop_o > 0.0 and self.is_training:
                    outputs = tf.layers.dropout(outputs, drop_o,
                                                noise_shape=[
                                                    bsz, 1, layer['filters']],
                                                training=self.is_training)
                inputs = outputs
            # Perform k max pooling
            # swap last two dimensions since top_k will be applied along the last dimension
            inputs = tf.transpose(inputs, [0, 2, 1])

            # extract top_k, returns two tensors [values, indices]
            inputs = tf.nn.top_k(inputs, k=self.k, sorted=True)[0]
            inputs = tf.reshape(inputs, [bsz, inputs.shape[1]*self.k])
            self.outputs = tf.layers.dense(
                inputs,
                units=self.num_class,
                activation=None,
                kernel_initializer=tf.glorot_uniform_initializer()
            )
            self.predictions = tf.argmax(self.outputs, axis=-1)
            self.out_prob = tf.nn.softmax(self.outputs)
        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.l2_reg_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.name)
