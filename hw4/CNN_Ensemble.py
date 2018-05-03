
# coding: utf-8

# In[2]:


import keras  
print(keras.__version__,', (should be 2.0.8)')  
import tensorflow
print(tensorflow.__version__,', (should be 1.4.0)') 


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import time
tStart = time.time()
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad


def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(math.floor(all_data_size * percentage))

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


def Parse_Data():
    df = pd.read_csv('train.csv')
    Y_train = df.label
    data = df.drop('label', axis =1 )
    data = data.feature
    X_train = data.str.split(" ",expand = True)

    df = pd.read_csv('test.csv')
    data = df.feature
    X_test = data.str.split(" ",expand = True)

    return X_train, Y_train, X_test


def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma
    
    
    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test


# In[16]:


X_train, Y_train, kX_test = Parse_Data()
#test_Parse_Data()

X_train = np.float32(X_train)/255
kX_test = np.float32(kX_test)/255
Y_train = np.int_(Y_train)

#normalize
X_train, kX_test = normalize(X_train, kX_test)

X_data = X_train
y_data = Y_train

# shape of data
X_data = X_data.reshape((X_data.shape[0], 48,48))
kX_test = kX_test.reshape((kX_test.shape[0], 48,48))
print(X_data.shape)    # (8 X 8) format
print(y_data.shape)


X_data = X_data.reshape((X_data.shape[0], X_data.shape[1], X_data.shape[2], 1))
kX_test = kX_test.reshape((kX_test.shape[0], kX_test.shape[1], kX_test.shape[2], 1))

# partition data into train/test sets
X_train, y_train, X_test, y_test = split_valid_set(X_data, y_data, 0.9)
X_train, y_train, X_train_2, y_train_2 = split_valid_set(X_train, y_train, 0.9)
# In[141]:
y_train = to_categorical(y_train)
y_train_2 = to_categorical(y_train_2)
Y_test = to_categorical(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#print('X_train:',X_train[0:5])


# In[17]:


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Merge, GlobalAveragePooling2D, Average, Maximum, Add, Concatenate, concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard


# In[18]:


def conv_pool_cnn(model_input):
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(7, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='conv_pool_cnn')
    
    return model
def conv_pool_cnn_batchnorm(model_input):
    
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(32, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu')(x)
    #x = Conv2D(7, (1, 1))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(7, activation='softmax')(x)
    #x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='conv_pool_cnn_batchnorm')
    model.summary()
    
    return model
def all_cnn(model_input):
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(7, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
        
    model = Model(model_input, x, name='all_cnn')
    model.summary()
    
    return model
def nin_cnn(model_input):
    
    
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
        
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(7, (1, 1))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='nin_cnn')
    model.summary()
    return model


# In[19]:


def compile_and_train(model, num_epochs, X_train, y_train, imageprocess = True): 
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc']) 
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=False, save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=128)
    #history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    
    batch_size = 128
    nb_epoch = 100
    #imageprocess = True
    if not imageprocess:
        history = model.fit(X_train, y_train, batch_size = batch_size, validation_split = 0.1, epochs = nb_epoch, verbose = 1)
    else:
        print('Using real-time data augmentation.')

        #should I normalize first? try samplewise norm!
        datagen = ImageDataGenerator(              #1
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,  
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range = 10, 
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            horizontal_flip=True, 
            vertical_flip=False)

        datagen.fit(X_train)                       

        history = model.fit_generator(datagen.flow(X_train, y_train,    
                            batch_size  = batch_size),
                            samples_per_epoch = X_train.shape[0],
                            epochs = nb_epoch,
                            steps_per_epoch = len(X_train),
                            validation_data = (X_test, Y_test),
                            callbacks=[checkpoint, tensor_board])
        
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['training', 'validation'], loc = 'upper left')
    plt.show()

    results = model.evaluate(X_test, Y_test)

    print('Test accuracy: ', results[1])
    return history

def evaluate_error(model):
    pred = model.predict(X_test, batch_size = 128)
    pred = np.argmax(pred, axis=1)
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]  
  
    return error


# In[20]:


input_shape = X_train[0,:,:,:].shape
print(input_shape)
model_input = Input(shape=input_shape)


# In[21]:


#for training

conv_pool_cnn_batchnorm_model = conv_pool_cnn_batchnorm(model_input)
_ = compile_and_train(conv_pool_cnn_batchnorm_model, num_epochs=100, X_train = X_train, y_train = y_train)
evaluate_error(conv_pool_cnn_batchnorm_model)

conv_pool_cnn_model = conv_pool_cnn(model_input)
_ = compile_and_train(conv_pool_cnn_model, num_epochs=100, X_train = X_train, y_train = y_train)
evaluate_error(conv_pool_cnn_model)

all_cnn_model = all_cnn(model_input)
_ = compile_and_train(all_cnn_model, num_epochs=100, X_train = X_train, y_train = y_train)
evaluate_error(all_cnn_model)

nin_cnn_model = nin_cnn(model_input)
_ = compile_and_train(nin_cnn_model, num_epochs=100, X_train = X_train, y_train = y_train)
evaluate_error(nin_cnn_model)


# In[22]:


def ensemble(models, model_input):
    
    outputs = [model.outputs[0] for model in models]
    y = Maximum()(outputs)
    model = Model(model_input, y, name='ensemble')
    
    return model


# In[23]:


#Load Weights
conv_pool_cnn_batchnorm_model = conv_pool_cnn_batchnorm(model_input)
conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)

conv_pool_cnn_batchnorm_model.load_weights('weights/conv_pool_cnn_batchnorm.98-0.47.hdf5')
conv_pool_cnn_model.load_weights('weights/conv_pool_cnn.99-0.22.hdf5')
all_cnn_model.load_weights('weights/all_cnn.98-0.28.hdf5')
nin_cnn_model.load_weights('weights/nin_cnn.99-1.13.hdf5')

models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model, conv_pool_cnn_batchnorm_model]

print(evaluate_error(conv_pool_cnn_batchnorm_model))
print(evaluate_error(conv_pool_cnn_model))
print(evaluate_error(all_cnn_model))
print(evaluate_error(nin_cnn_model))


# In[25]:


index = 1 
for m in models:
    pred = m.predict(kX_test)
    y_ = pred.argmax(axis = -1)
    output_path = 'pred_'+str(index)+'.csv'
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i, v))
    index+=1
    print(index)


# In[12]:


ensemble_model = ensemble(models, model_input)
print(evaluate_error(ensemble_model))
#results = ensemble_model.evaluate(X_test, y_test)
#print('Test accuracy: ', results[1])
pred = ensemble_model.predict(kX_test)
y_ = pred.argmax(axis = -1)
output_path = 'en_output.csv'
with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(y_):
        f.write('%d,%d\n' %(i, v))

