
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys

# In[2]:


def normalize(X_all):
    # Feature normalization with train and test X
    mu = np.loadtxt('mu')
    mu.astype('float32')
    sigma = np.loadtxt('sigma')
    sigma.astype('float32')
    print(mu)
    print(sigma)
    
    X_train_test = X_all
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    print('type:',X_train_test.dtype)
    X_train_test_normed = (X_train_test - mu) / sigma
   
    X_all = X_train_test_normed
    return X_all
def Parse_Testing_Data():
    test_path = sys.argv[1]
    df = pd.read_csv(test_path)
    #Y_train = df.label
    data = df.feature
    X_test = data.str.split(" ",expand = True)

    print(len(X_test))
    print('x_test',X_test.shape)
    return X_test


# In[3]:


from keras.models import load_model

X_test = Parse_Testing_Data()
X_test = np.float32(X_test)/255
X_test = normalize(X_test)
X_test = X_test.reshape((X_test.shape[0],48,48))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
model_path = 'best.h5'
model = load_model(model_path)
y_pred = model.predict(X_test)

y_pred = y_pred.argmax(axis = -1)
output_path = sys.argv[2]
#output_path = 'output.csv'
with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(y_pred):
        f.write('%d,%d\n' %(i, v))

