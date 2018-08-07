import tensorflow as tf


@tf.custom_gradient
def apply_custom_lr(x, lr):
    def grad(dy):
        return dy * lr, 0
    return x, grad


if __name__ == '__main__':
    tf.set_random_seed(42)
    N = 10
    x = tf.random_normal([N, 5], 0, 1)
    y = tf.random_normal([N, 1], 0, 2)
    W = tf.Variable(tf.random_normal([5, 1], 0, 1))
    W1 = tf.Variable(tf.random_normal([5, 1], 0, 1))
    copy_W = W1.assign(W)
    b = tf.Variable(tf.ones(1))
    b1 = tf.Variable(tf.ones(1))
    W1 = apply_custom_lr(W1, 0.5)
    b1 = apply_custom_lr(b1, 0.5)
    y_pred = tf.nn.xw_plus_b(x, W, b)
    y1_pred = tf.nn.xw_plus_b(x, W1, b1)
    loss = tf.losses.mean_squared_error(y, y_pred)
    loss1 = tf.losses.mean_squared_error(y, y1_pred)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    grads_and_vars = optimizer.compute_gradients(
        loss) + optimizer.compute_gradients(loss1)
    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(copy_W)
        print(sess.run(W1))
        for _ in range(10):
            _, l, l1 = sess.run([train_op, loss, loss1])
            print('Loss', l)
            print('Loss 0.5', l1)
            # print('Gradients', gv)
