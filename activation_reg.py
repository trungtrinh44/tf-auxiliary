import tensorflow as tf


def AR(x, alpha):
    return alpha * tf.reduce_mean(tf.square(x))


def TAR(x, beta):
    t = x[1:] - x[:-1]
    return beta * tf.reduce_mean(tf.square(t))


if __name__ == '__main__':
    tf.set_random_seed(42)
    x = tf.random_normal([5, 2])
    s = TAR(x, 0.5)
    with tf.Session() as sess:
        x, s = sess.run([x, s])
        print(x)
        print(s)
