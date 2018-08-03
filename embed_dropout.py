import tensorflow as tf


def embedding_dropout(embed, dropout, reuse=None, name='embedding_dropout'):
    with tf.variable_scope(name, reuse=reuse):
        keep_prob = tf.convert_to_tensor(
            1-dropout, dtype=tf.float32, name='keep_prob')
        shape = tf.shape(embed)
        mask = tf.random_uniform(shape=(shape[0], 1), dtype=tf.float32)
        mask += keep_prob
        mask = tf.floor(mask)
        result = embed * mask
        result /= keep_prob
    return result


if __name__ == '__main__':
    import numpy as np
    V = 50
    h = 4
    bptt = 10
    batch_size = 2
    words = np.random.random_integers(low=0, high=V-1, size=(batch_size, bptt))
    with tf.Session() as sess:
        embed = tf.Variable(tf.random_uniform(
            [V, h], -1.0, 1.0), name="W")
        words = tf.convert_to_tensor(words, dtype=tf.int32)
        embed_drop = tf.nn.embedding_lookup(
            embedding_dropout(embed, dropout=0.1), words
        )
        embed = tf.nn.embedding_lookup(embed, words)
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        print(sess.run(embed))
        print(sess.run(embed_drop))
        print(sess.run(embed_drop))
