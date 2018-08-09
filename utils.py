import numpy as np
import logging

def get_batch(source, bptt, i, evaluate=False):
    if evaluate:
        seq_len = bptt
    else:
        real_bptt = bptt if np.random.random() < 0.95 else bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(real_bptt, 5)))
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target


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
