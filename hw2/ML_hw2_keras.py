
# coding: utf-8

# In[150]:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
from math import log, floor
import sys

# In[151]:


def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train_cont = X_train[['age','capital_gain','capital_loss','hours_per_week','fnlwgt']]
    X_train_cont = np.array(X_train_cont)
   # X_train_cont = single_normalize(X_train_cont)
    X_train = X_train.drop(['age','capital_gain','capital_loss','hours_per_week','fnlwgt','occupation_?','native_country_?','workclass_?','occupation_Other-service','race_Other'], axis = 1)
    X_train = np.array(X_train.values)
    X_train = X_train * 0.8


    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)

    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test_cont = X_test[['age','capital_gain','capital_loss','hours_per_week','fnlwgt']]
    X_test_cont = np.array(X_test_cont)
    (X_train_cont,X_test_cont) = normalize(X_train_cont,X_test_cont)
    X_test = X_test.drop(['age','capital_gain','capital_loss','hours_per_week','fnlwgt','occupation_?','native_country_?','workclass_?','occupation_Other-service','race_Other'], axis = 1)
    X_test = np.array(X_test.values)
    X_test = X_test * 0.8
    
    X_test = np.concatenate((X_test,X_test_cont),axis=1)
    X_train = np.concatenate((X_train,X_train_cont),axis=1)
    return (X_train, Y_train, X_test)

def single_normalize(X_all):
    # Feature normalization with train and test X
    X_train_test = X_all
    mu = (sum(X_train_test) / X_train_test.shape[0])
  #  print('shape of mu',np.shape(mu))
    sigma = np.std(X_train_test, axis=0)
 #   print('shape of sigma',np.shape(sigma))
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed
#    print("X_norm=",X_all)
    return X_all
def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
  #  print('shape of mu',np.shape(mu))
    sigma = np.std(X_train_test, axis=0)
 #   print('shape of sigma',np.shape(sigma))
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
#    print("X_norm=",X_all)
    return X_all, X_test
def f_single_normalize(X_all):
    # Feature normalization with train and test X
    X_train_test = X_all
    xmax = np.amax(X_train_test, axis = 0)
    xmin = np.amin(X_train_test, axis = 0)
    xmax = np.tile(xmax, (X_train_test.shape[0], 1))
    xmin = np.tile(xmin, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - xmin) / (xmax - xmin)

    X_all = X_train_test_normed
    return X_all

def f_normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    xmax = np.amax(X_train_test, axis = 0)
    xmin = np.amin(X_train_test, axis = 0)
    print('xmax shape:',np.shape(xmax))
    xmax = np.tile(xmax, (X_train_test.shape[0], 1))
    xmin = np.tile(xmin, (X_train_test.shape[0], 1))
    #sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - xmin) / (xmax - xmin)

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    print(X_all)
    return X_all, X_test

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[152]:


batch_size = 32
nb_classes = 2
nb_epoch = 100
percent = 0.1


# In[153]:


#(X_train, Y_train, X_test) = load_data('feature/X_train','feature/Y_train','feature/X_test')
(X_train, Y_train, X_test) = load_data(sys.argv[1],sys.argv[2],sys.argv[3])
#print(Y_train)
Y_train = to_categorical(Y_train, num_classes=nb_classes)
print('X_S',np.shape(X_train))
print('Y_S',np.shape(Y_train))
#print('X_test=',X_test)
#print(Y_train)

#(X_train, X_test) = f_normalize(X_train, X_test)

(X_train, Y_train, X_valid, Y_valid) = split_valid_set(X_train,Y_train,percent)
#print('X_test=',X_test)



# In[ ]:


# Logistic regression model
model = Sequential()

earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 10, verbose = 1, mode = 'max')

#model.add(Dropout(0.5,input_shape=(118,),))

#model.add(Dense(output_dim=40, init='normal', activation='sigmoid'))

model.add(Dense(output_dim = 40, input_shape=(118,), init='normal', activation='sigmoid'))

model.add(Dense(output_dim=22,  activation='sigmoid'))

#model.add(Dropout(0.5))
#model.add(Dense(output_dim=22,  activation='sigmoid'))

#model.add(Dense(output_dim=12,  activation='relu'))
#model.add(Dropout(0.5))
"""
model.add(Dense(output_dim=300,  activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(output_dim=600,  activation='relu'))
#model.add(Dropout(0.5))
"""
model.add(Dense(output_dim=2, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_s = model.summary()


# In[ ]:


# Train
#history = model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size, verbose=1,callbacks = [earlyStopping],shuffle = True,validation_data = (X_valid, Y_valid))
history = model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size, verbose=1)


# In[ ]:


# Evaluate
evaluation = model.evaluate(X_valid, Y_valid, verbose=1)
print('Summary: Loss over the test dataset: %.4f, Accuracy: %.4f' % (evaluation[0], evaluation[1]))


# In[ ]:


result = model.predict(X_test,batch_size = batch_size,verbose =1 )
#result_train = model.predict(X_train,batch_size = batch_size,verbose =1 )


# In[ ]:


#print(result)


# In[ ]:


#print(result[:,1])
#print(result[1:50,1])
print(np.shape(result))
y=np.zeros(np.shape(result)[0])
for i in range(np.shape(result)[0]):
    #print(i)
    if(result[i,1]>result[i,0]):
        y[i]=1
    else:
        y[i]=0
#print(y)


output_path = sys.argv[4]
with open(output_path,'w+') as f:
    f.write('id,label\n')
    for i,v in enumerate(y):
        f.write('%d,%d\n' %(i+1, v))

print('writed')

# In[ ]:

"""
with open('log.txt','a+') as log:
    log.write(model_s)
    log.write('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
    log.write('------------------------------------------------------------------------------------------------')
"""
