import tensorflow as tf

def AR(x, alpha):
    return alpha * tf.nn.l2_loss(x)

def TAR(x, alpha):
    t = x[1:] - x[:-1]
    return alpha * tf.nn.l2_loss(t),t

if __name__ == '__main__':
    tf.set_random_seed(42)
    x = tf.random_normal([5,1])
    s, t = TAR(x, 0.5)
    with tf.Session() as sess:
        x, s, t = sess.run([x, s, t])
        print(x)
        print(s)
        print(t)