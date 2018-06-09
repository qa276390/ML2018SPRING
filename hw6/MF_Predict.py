
# coding: utf-8

# In[8]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import math
import pandas as pd
#import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
#from CFModel import CFModel, DeepModel, CFModel_Bias
import tensorflow as tf
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

import keras
from keras.layers import Input, Add, Dot, Embedding, Flatten, Dense
from keras.models import Model
from keras.layers.merge import add, dot, concatenate, multiply, average
import sys
#import sklearn


# In[9]:
test_file = sys.argv[1]
pred_path = sys.argv[2]

#RATINGS_CSV_FILE = 'data/train.csv'
MODEL_WEIGHTS_FILE = 'mf_weights.h5'
K_FACTORS = 128
RNG_SEED = 1446557
std = 1
mean = 0


# In[10]:


std = np.load('data/std.npy')
mean = np.load('data/mean.npy')


# In[11]:


def Model_Bias(n_users, m_items, k):
    user = Input(shape=(1,))
    movie = Input(shape=(1,))
    
    user_w = Embedding(n_users, k)(user)
    user_flat = Flatten()(user_w)
    movie_w = Embedding(m_items, k)(movie)
    movie_flat = Flatten()(movie_w)
    
    movie_b = Embedding(m_items, 1)(movie)
    movie_b_flat = Flatten()(movie_b)
    user_b = Embedding(n_users, 1)(user)
    user_b_flat = Flatten()(user_b)
    
    ratings = Dot(axes = -1)([user_flat, movie_flat])
    
   
    output = add(inputs=[movie_b_flat, user_b_flat, ratings])
   
    return Model(inputs = [user, movie], outputs = output)


# In[12]:


#ratings = pd.read_csv(RATINGS_CSV_FILE, usecols=['UserID', 'MovieID', 'Rating'])
#max_userid = ratings['UserID'].drop_duplicates().max()
#max_movieid = ratings['MovieID'].drop_duplicates().max()
#print(len(ratings), 'ratings loaded.')


# In[13]:


#trained_model = Model_Bias(max_userid, max_movieid, K_FACTORS)
#trained_model.load_weights(MODEL_WEIGHTS_FILE)
trained_model = keras.models.load_model(MODEL_WEIGHTS_FILE)


# In[14]:


test = pd.read_csv(test_file)
submit = pd.read_csv('data/SampleSubmisson.csv')
print('Predicting...')
submit['Rating'] = test.apply(lambda x: trained_model.predict([np.array([x['UserID']]),np.array([x['MovieID']])])[0][0] * std + mean, axis=1)
print(submit.head)

submit.to_csv(pred_path, index = False)

