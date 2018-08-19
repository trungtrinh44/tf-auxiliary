
# coding: utf-8

# In[1]:


from data import Corpus
import trainer
from utils import batchify
import json
import tensorflow as tf
import numpy as np
np.random.seed(42)
tf.set_random_seed(42)

# In[2]:


# corpus = Corpus('baomoi/')


# In[3]:


# In[4]:


# np.save('baomoi/train.npy', corpus.train)


# In[5]:


# np.save('baomoi/valid.npy', corpus.valid)


# In[6]:


# np.save('baomoi/test.npy', corpus.test)


# In[7]:


# with open('word2idx.json','w') as out:
#     json.dump(corpus.dictionary.word2idx, out)


# In[8]:


with open('baomoi_punc/word2idx.json','r') as inp:
    word2idx = json.load(inp)


# In[9]:


with open('baomoi_punc/train.npy','rb') as inp:
    train = np.load(inp)
with open('baomoi_punc/test.npy','rb') as inp:
    test = np.load(inp)
with open('baomoi_punc/valid.npy','rb') as inp:
    valid = np.load(inp)


# In[10]:


train_data = batchify(train, 202).T
val_data = batchify(valid, 158).T


# In[11]:


VERSION = 7
params = dict(
    model_configs = {
   "rnn_layers":[
          {
             "units": 1200,
             "input_size":400,
             "drop_i": 0.01,
             "wdrop": 0.02,
             "drop_o": 0.01
          },
          {
             "units": 1200,
             "input_size": 1200,
             "wdrop": 0.02,
             "drop_o": 0.01
          },
          {
             "units": 400,
             "input_size": 1200,
             "drop_o": 0.1,
             "wdrop": 0.02
          }
       ],
       "vocab_size": len(word2idx),
       "drop_e":0.0
    },
    optimizer = tf.train.GradientDescentOptimizer,
    wdecay = 1.2e-6,
    alpha = 0.0,
    beta = 0.0,
    clip_norm = 0.3,
    bptt = 200,
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


# In[ ]:

lr = 4.0
for _ in range(10000):
    my_trainer.train_dev_loop(train_data, val_data, lr)
    lr = lr / np.sqrt(2.0)
