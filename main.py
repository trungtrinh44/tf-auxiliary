from data import Corpus
import trainer
from utils import batchify
import json
import tensorflow as tf
import numpy as np


with open('wiki/word2idx.json', 'r') as inp:
    word2idx = json.load(inp)
with open('wiki/char2idx.json', 'r') as inp:
    char2idx = json.load(inp)


VERSION = 102
params = dict(
    model_configs = {
      "rnn_layers":[
          {
             "units": 1024,
             "drop_i": 0.2,
             "wdrop": 0.4,
             "drop_o": 0.2
          },
          {
             "units": 1024,
             "wdrop": 0.4,
             "drop_o": 0.2
          },
          {
             "units": 1024,
             "drop_o": 0.2,
             "wdrop": 0.4
          }
       ],
       "vocab_size": len(word2idx) + 1,
       "drop_e": 0.4,
       "char_vocab_size": len(char2idx) + 1,
       "char_cnn_options": {
           "layers": [
                [1, 16],
                [2, 16],
                [3, 32],
                [4, 64],
                [5, 128],
                [6, 256],
                [7, 512]
           ],
           "n_highways": 2
       },
       "char_vec_size": 16,
       "projection_dims": 512,
       "skip_connection": True
    },
    optimizer = tf.train.GradientDescentOptimizer,
    negative_samples = 10240,
    wdecay = 1.2e-6,
    alpha = 1e-6,
    beta = 1e-6,
    clip_norm = 0.3,
    bptt = 70,
    use_ema = True,
    save_freq = 1000,
    log_path = '{}/logs'.format(VERSION),
    train_summary_dir = '{}/train_summary/'.format(VERSION),
    test_summary_dir = '{}/test_summary/'.format(VERSION),
    checkpoint_dir = '{}/checkpoints/'.format(VERSION)
)


# In[12]:


my_trainer = trainer.Trainer(**params)

my_trainer.logger.info('Trainer params {}'.format(params))
# In[13]:


tf.reset_default_graph()
my_trainer.build()

with open('wiki/train_word.npy','rb') as inp:
    train_word = np.load(inp)
with open('wiki/train_char.npy','rb') as inp:
    train_char = np.load(inp)
with open('wiki/valid_word.npy','rb') as inp:
    valid_word = np.load(inp)
with open('wiki/valid_char.npy','rb') as inp:
    valid_char = np.load(inp)

train_word = batchify(train_word, 81).T
train_char = batchify(train_char, 81).T
valid_word = batchify(valid_word, 72).T
valid_char = batchify(valid_char, 72).T

lr = 10.0
for _ in range(10000):
    my_trainer.train_dev_loop(train_word, train_char, valid_word, valid_char, lr)
    lr = max(lr / np.sqrt(2.0), 2.0)
