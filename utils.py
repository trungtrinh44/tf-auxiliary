import numpy as np


def get_batch(source, bptt, i):
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
