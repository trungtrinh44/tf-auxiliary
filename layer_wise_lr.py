# -*- coding: utf-8 -*-
"""
@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import tensorflow as tf


def apply_custom_lr(lr):
    @tf.custom_gradient
    def child(x):
        def grad(dy):
            return dy * lr
        return x, grad
    return child


if __name__ == '__main__':
    tf.set_random_seed(42)
    import numpy as np
    N = 10
    x = np.random.normal(0, 1, [N, 5])
    x = tf.constant(x, dtype=tf.float32)
    y = np.random.normal(0, 2, [N, 1])
    y = tf.constant(y, dtype=tf.float32)
    W = tf.Variable(tf.random_normal([5, 1], 0, 1))
    W1 = tf.Variable(tf.random_normal([5, 1], 0, 1))
    copy_W = W1.assign(W)
    b = tf.Variable(tf.ones(1))
    b1 = tf.Variable(tf.ones(1))
    W1 = apply_custom_lr(0.0)(W1)
    b1 = apply_custom_lr(0.0)(b1)
    y_pred = tf.nn.xw_plus_b(x, W, b)
    y1_pred = tf.nn.xw_plus_b(x, W1, b1)
    loss = tf.losses.mean_squared_error(y, y_pred)
    loss1 = tf.losses.mean_squared_error(y, y1_pred)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    grads_and_vars = optimizer.compute_gradients(loss) + optimizer.compute_gradients(loss1)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    grads_5 = tf.gradients(loss1, [W1, b1])
    grads = tf.gradients(loss, [W, b])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(copy_W)
        print(sess.run(W1))
        for _ in range(10):
            _, l, l1, *gv = sess.run([train_op, loss, loss1] + grads_5 + grads)
            print('Loss', l)
            print('Loss 0.5', l1)
            print('Grads', gv)