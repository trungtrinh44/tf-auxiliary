import tensorflow as tf
import numpy as np

arr = tf.TensorArray(dtype=tf.float32, size=10, dynamic_size=True)
seq_lens = [3, 5, 7]
i = tf.constant(0)
values = tf.constant(np.random.rand(7, 3, 4), dtype=tf.float32)
W = tf.get_variable(name='W', shape=(4, 2), dtype=tf.float32)
test_values = tf.reshape(values, (21, 4)) @ W
test_values = tf.reshape(test_values, [7, 3, 2])
mask = tf.expand_dims(tf.transpose(tf.sequence_mask(seq_lens, dtype=tf.float32), (1, 0)), axis=-1)
max_test_values = test_values * mask + (1-mask)*-1e6
max_test_values = tf.reduce_max(max_test_values, axis=0)
mean_test_values = test_values * mask
mean_test_values = tf.reduce_sum(mean_test_values, axis=0) / tf.expand_dims(tf.to_float(seq_lens), axis=-1)
bptt = tf.constant(2)
max_len = tf.reduce_max(seq_lens)
max_val = tf.constant(-1e6, shape=(3, 2))
mean_val = tf.constant(0.0, shape=(3, 2))


def cond(i, inputs, sl, max_val, mean_val, bptt, max_len): return i < max_len


def body(i, inputs, sl, max_val, mean_val, bptt, max_len):
    i_to = tf.minimum(i+bptt, max_len)
    slice_inputs = inputs[i:i_to]
    s = tf.shape(slice_inputs)
    slice_inputs = tf.reshape(slice_inputs, [s[0]*s[1], s[2]])
    outputs = slice_inputs @ W
    outputs = tf.reshape(outputs, [s[0], s[1], -1])
    mask = tf.expand_dims(tf.transpose(tf.sequence_mask(tf.minimum(sl-i, bptt), dtype=tf.float32), (1, 0)), axis=-1)
    max_outputs = outputs * mask + (1 - mask) * -1e6
    max_val = tf.maximum(max_val, tf.reduce_max(max_outputs, axis=0))
    mean_outputs = outputs * mask
    mean_val = (mean_val * tf.expand_dims(tf.to_float(tf.minimum(i, sl)), axis=-1) + tf.reduce_sum(mean_outputs, axis=0)) / tf.expand_dims(tf.to_float(tf.minimum(i_to, sl)), axis=-1)
    return i_to, inputs, sl, max_val, mean_val, bptt, max_len


i_to, inputs, sl, max_val, mean_val, bptt, max_len = tf.while_loop(cond, body, [i, values, seq_lens, max_val, mean_val, bptt, max_len])
max_loop_grad = tf.gradients(max_val, [W])[0]
max_test_grad = tf.gradients(max_test_values, [W])[0]
mean_loop_grad = tf.gradients(mean_val, [W])[0]
mean_test_grad = tf.gradients(mean_test_values, [W])[0]
sess = tf.Session()
sess.run(tf.global_variables_initializer())
a = sess.run([max_val, max_loop_grad, mean_val, mean_loop_grad])
b = sess.run([max_test_values, max_test_grad, mean_test_values, mean_test_grad])
print([x-y for x, y in zip(a, b)])
