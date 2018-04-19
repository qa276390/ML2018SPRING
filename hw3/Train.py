


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.utils.np_utils import to_categorical
import time
tStart = time.time()


def Model2():
    filt_size = (3, 3)
    model2 = Sequential()
    model2.add(Convolution2D(32, filt_size, input_shape=(48,48,1), activation='relu', padding='same'))
    model2.add(Convolution2D(32, filt_size, activation='relu', padding='same'))
    model2.add(BatchNormalization())
    model2.add(MaxPooling2D((2,2)))
    model2.add(Dropout(0.1))
    
    model2.add(Convolution2D(64, filt_size, activation='relu', padding='same'))
    model2.add(Convolution2D(64, filt_size, activation='relu', padding='same'))
    model2.add(BatchNormalization())
    model2.add(MaxPooling2D((2,2)))
    model2.add(Dropout(0.3))
    
    model2.add(Convolution2D(128, filt_size, activation='relu', padding='same'))
    model2.add(Convolution2D(128, filt_size, activation='relu', padding='same'))
    model2.add(Convolution2D(128, filt_size, activation='relu', padding='same'))
    model2.add(BatchNormalization())
    model2.add(MaxPooling2D((2,2)))
    model2.add(Dropout(0.4))
    
    model2.add(Convolution2D(256, filt_size, activation='relu', padding='same'))
    model2.add(Convolution2D(256, filt_size, activation='relu', padding='same'))
    #model2.add(Convolution2D(256, filt_size, activation='relu', padding='same'))
    model2.add(BatchNormalization())
    model2.add(MaxPooling2D((2,2)))
    model2.add(Dropout(0.5))
    
    
    model2.add(Flatten())
    model2.add(Dense(1024, activation='relu'))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.5))
    model2.add(Dense(1024, activation='relu'))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.5))
    model2.add(Dense(7))
    model2.add(Activation('softmax'))
    return model2

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)

    return (X[randomize], Y[randomize])
def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(math.floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


def Parse_Data():
    path = sys.argv[1]
    df = pd.read_csv(path)
    Y_train = df.label
    data = df.drop('label', axis =1 )
    data = data.feature
    X_train = data.str.split(" ",expand = True)

    print(len(X_train))
    print(X_train.shape)
    print(Y_train.shape)
    print(len(Y_train))
    X_train.to_csv("X_train",index=None)
    Y_train.to_csv("Y_train",index=None)




def test_Parse_Data():
    df = pd.read_csv('test.csv')
    #Y_train = df.label
    data = df.feature
    X_test = data.str.split(" ",expand = True)

    print(len(X_test))
    print('x_test',X_test.shape)
    X_test.to_csv("X_test",index=None)



def normalize(X_all, X_test):
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


# In[2]:


Parse_Data()
test_Parse_Data()
X_train = pd.read_csv('X_train')
Y_train = pd.read_csv('Y_train',header=None)
kX_test = pd.read_csv('X_test')

X_train = np.float32(X_train)/255
kX_test = np.float32(kX_test)/255
Y_train = np.int_(Y_train)

#normalize
X_train, kX_test = normalize(X_train, kX_test)

# In[135]:


X_data = X_train
y_data = Y_train



X_data = X_data.reshape((X_data.shape[0], 48,48))
kX_test = kX_test.reshape((kX_test.shape[0], 48,48))
print(X_data.shape)    # (8 X 8) format
print(y_data.shape)


X_data = X_data.reshape((X_data.shape[0], X_data.shape[1], X_data.shape[2], 1))
kX_test = kX_test.reshape((kX_test.shape[0], kX_test.shape[1], kX_test.shape[2], 1))


y_data = to_categorical(y_data)



# partition data into train/test sets
X_train, y_train, X_test, y_test = split_valid_set(X_data, y_data, 0.9)

# In[141]:



print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[3]:

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


#model = VGG_16()
model = Model2()
model.summary()
#adam = optimizers.Adam(lr = 0.01)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


batch_size = 128
nb_epoch = 100
imageprocess = False
if not imageprocess:
    history = model.fit(X_train, y_train, batch_size = 128, validation_split = 0.1, epochs = 100, verbose = 1)
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

    history = model.fit_generator(datagen.flow(X_train, y_train,batch_size  = batch_size),
                        samples_per_epoch = X_train.shape[0],
                        epochs = nb_epoch,
                        steps_per_epoch = len(X_train),
                        validation_data = (X_test, y_test))


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()

results = model.evaluate(X_test, y_test)

print('Test accuracy: ', results[1])


# In[ ]:


model.model.save("my_model.h5")


# In[ ]:


pred = model.predict(kX_test)
y_ = pred.argmax(axis = -1)
output_path = 'output.csv'
with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(y_):
        f.write('%d,%d\n' %(i, v))


# In[ ]:


tEnd = time.time()
print ("It cost %f sec" % (tEnd - tStart))
print (tEnd - tStart)


