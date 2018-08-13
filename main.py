
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


import numpy as np


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


with open('word2idx.json','r') as inp:
    word2idx = json.load(inp)


# In[9]:


with open('baomoi/train.npy','rb') as inp:
    train = np.load(inp)
with open('baomoi/test.npy','rb') as inp:
    test = np.load(inp)
with open('baomoi/valid.npy','rb') as inp:
    valid = np.load(inp)


# In[10]:


train_data = batchify(train, 433).T
val_data = batchify(valid, 485).T


# In[11]:


VERSION = 3
params = dict(
    model_configs = {
   "rnn_layers":[
          {
             "units": 1200,
             "input_size":400,
             "drop_i":0.65,
             "drop_w":0.5,
             "drop_o":0.3
          },
          {
             "units": 1200,
             "input_size": 1200,
             "drop_w":0.5,
             "drop_o":0.3
          },
          {
             "units":400,
             "input_size": 1200,
             "drop_o":0.4,
             "drop_w":0.5
          }
       ],
       "vocab_size": len(word2idx),
       "drop_e":0.1
    },
    optimizer = tf.train.GradientDescentOptimizer,
    learning_rate = 10.0,
    decay_freq = 40000,
    decay_rate = 0.5,
    alpha = 2.0,
    beta = 1.0,
    clip_norm = 0.25,
    bptt = 100,
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


my_trainer.session.run(tf.global_variables_initializer())
for _ in range(10000):
    my_trainer.train_dev_loop(train_data, val_data)
