
# coding: utf-8

# In[5]:


from __future__ import print_function
import sys
import keras
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
print('keras:',keras.__version__)

from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
print('np:',np.__version__)
import time

import tensorflow as tf
print('tf:',tf.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)




# load images
train_num = 130000
#X = np.load('image.npy')
img_path = sys.argv[1]
X = np.load(img_path)
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), -1))
x_train = X[:train_num]
x_val = X[train_num:]
x_train.shape, x_val.shape




encoder = load_model('encoder.h5')


# In[10]:


# after training, use encoder to encode image, and feed it into Kmeans
encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)


# In[11]:


# get test cases
test_path = sys.argv[2]
#f = pd.read_csv('test_case.csv')
f = pd.read_csv(test_path)
print('shape=',f.shape)
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])


# In[12]:


# predict
pred_path = sys.argv[3]
#o = open('prediction.csv', 'w')
o = open(pred_path, 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1  # two images in same cluster
    else: 
        pred = 0  # two images not in same cluster
    o.write("{},{}\n".format(idx, pred))
o.close()


# In[ ]:



