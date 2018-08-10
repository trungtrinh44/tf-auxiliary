import numpy as np
import logging


def get_batch(source, bptt, evaluate=False, inference=False):
    def generator():
        i = 0
        while i < len(source) - 1:
            if evaluate:
                seq_len = bptt
            else:
                real_bptt = bptt if np.random.random() < 0.95 else bptt / 2.
                # Prevent excessively small or negative sequence lengths
                seq_len = min(
                    max(5, int(np.random.normal(real_bptt, 5))), int(1.2*bptt))
            seq_len = min(seq_len, len(source) - 1 - i)
            data = source[i:i+seq_len]
            if inference:
                yield data, np.array([seq_len]*source.shape[1]), i == 0
            else:
                target = source[i+1:i+1+seq_len]
                yield data, target, np.array([seq_len]*source.shape[1]), i == 0
            i += seq_len
    return generator


def batchify(source, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(source) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = source[:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    data = np.reshape(data, [bsz, nbatch])
    return data


def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger


if __name__ == '__main__':
    import tensorflow as tf
    import time
    X = np.random.randint(0, 10, [1000, 100])
    gen = get_batch(X, 100)
    inp = tf.placeholder(dtype=tf.int32, shape=[None, None])
    with tf.Session() as sess:
        for i in gen():
            s = time.time()
            v = sess.run(inp, {inp: i[0]})
            print(time.time()-s)
