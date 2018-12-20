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
test_values = test_values * mask + (1-mask)*-1e6
test_values = tf.reduce_max(test_values, axis=0)
bptt = tf.constant(2)
max_len = tf.reduce_max(seq_lens)
val = tf.constant(-1e6, shape=(3, 2))


def cond(i, inputs, sl, val, bptt, max_len): return i < max_len


def body(i, inputs, sl, val, bptt, max_len):
    i_to = tf.minimum(i+bptt, max_len)
    slice_inputs = inputs[i:i_to]
    s = tf.shape(slice_inputs)
    slice_inputs = tf.reshape(slice_inputs, [s[0]*s[1], s[2]])
    outputs = slice_inputs @ W
    outputs = tf.reshape(outputs, [s[0], s[1], -1])
    mask = tf.expand_dims(tf.transpose(tf.sequence_mask(tf.minimum(sl-i, bptt), dtype=tf.float32), (1, 0)), axis=-1)
    outputs = outputs * mask + (1 - mask) * -1e6
    val = tf.maximum(val, tf.reduce_max(outputs, axis=0))
    return i_to, inputs, sl, val, bptt, max_len


i_to, inputs, sl, val, bptt, max_len = tf.while_loop(cond, body, [i, values, seq_lens, val, bptt, max_len])
loop_grad = tf.gradients(val, [W])
test_grad = tf.gradients(test_values, [W])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([val, loop_grad]))
print(sess.run([test_values, test_grad]))
