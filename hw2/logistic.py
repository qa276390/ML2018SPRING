import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

# If you wish to get the same shuffle result
# np.random.seed(2401)

def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train_cont = X_train[['fnlwgt','age','capital_gain','capital_loss','hours_per_week']]
    col_cont = X_train_cont.columns
    #X_train_cont = X_train[['age','capital_gain','capital_loss','hours_per_week']]
    X_train_cont = np.array(X_train_cont)
    #X_train_cont = f_single_normalize(X_train_cont)
    X_train = X_train.drop(['race_Other','age','capital_gain','capital_loss','hours_per_week','fnlwgt','occupation_?','native_country_?','workclass_?','occupation_Other-service'], axis = 1)
    col = X_train.columns
    X_train = np.array(X_train.values)
    #X_train = X_train * 0.5


    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)

    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test_cont = X_test[['fnlwgt','age','capital_gain','capital_loss','hours_per_week']]
    #X_test_cont = X_test[['age','capital_gain','capital_loss','hours_per_week']]
    X_test_cont = np.array(X_test_cont)
    
    (X_train_cont, X_test_cont) = normalize(X_train_cont, X_test_cont)
    X_test = X_test.drop(['race_Other','age','capital_gain','capital_loss','hours_per_week','fnlwgt','occupation_?','native_country_?','workclass_?','occupation_Other-service'], axis = 1)
    X_test = np.array(X_test.values)
    #X_test = X_test * 0.5
    X_test = np.concatenate((X_test,X_test_cont),axis=1)
    
    X_train = np.concatenate((X_train,X_train_cont),axis=1)

    
    #col = np.concatenate((col,col_cont),axis=1)
    """
    with open('./logistic_params/_w','w+') as f:
        #f.write('id,label\n')
        k=0
        for i,v in enumerate(col):
            f.write('%s\n' %(v))
            k=i
        k+=1
        for i,v in enumerate(col_cont):
            f.write('%s\n' %(v))
    """
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
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    #print('X shape:',np.shape(X))
    #print('Y shape:',np.shape(Y))

    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    print('shape of mu',np.shape(mu))
    sigma = np.std(X_train_test, axis=0)
    print('shape of sigma',np.shape(sigma))
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    print("X_norm=",X_all)
    return X_all, X_test

def f_normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    xmax = np.amax(X_train_test,axis=0)
    xmin = np.amin(X_train_test,axis=0)
    xmax = np.tile(xmax, (X_train_test.shape[0], 1))
    xmin = np.tile(xmin, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - xmin) / (xmax - xmin)

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def f_single_normalize(X_all):
    # Feature normalization with train and test X
    X_train_test = X_all
    xmax = np.amax(X_train_test,axis=0)
    xmin = np.amin(X_train_test,axis=0)
    xmax = np.tile(xmax, (X_train_test.shape[0], 1))
    xmin = np.tile(xmin, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - xmin) / (xmax - xmin)

    # Split to train, test again
    X_all = X_train_test_normed
    return X_all

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return

def train(X_all, Y_all, save_dir):

    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

    # Initiallize parameter, hyperparameter
    w = np.zeros((118,))
    b = np.zeros((1,))
    l_rate = 0.1
    batch_size = 32
    print('lr={},batch={}'.format(l_rate,batch_size))
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 1000
    save_param_iter = 50
    reg = 0.01

    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w'), w)
            np.savetxt(os.path.join(save_dir, 'b'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)

        # Random shuffle
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            reg_loss = 0.5 * reg *np.sum(w**2)
            #total_loss += (cross_entropy + reg_loss)
            total_loss += (cross_entropy)

            #w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0) + w * reg
            w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0) 
            b_grad = np.mean(-1 * (np.squeeze(Y) - y))

            # SGD updating parameters
            w = w - l_rate * w_grad
            b = b - l_rate * b_grad

        
    return

def infer(X_test, save_dir, output_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_dir)
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #output_path = os.path.join(output_dir, 'log_prediction.csv')
    """
    output_path = output_dir
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return

def main(opts):
    # Load feature and label
    X_all, Y_all, X_test = load_data(opts.train_data_path, opts.train_label_path, opts.test_data_path)
   
    #X_all, Y_all, X_test = collect_data(X_all, Y_all, X_test)
   
   # Normalization
   # X_all, X_test = f_normalize(X_all, X_test)


    # To train or to infer
    if opts.train:
        train(X_all, Y_all, opts.save_dir)
    elif opts.infer:
        infer(X_test, opts.save_dir, opts.output_dir)
    else:
        print("Error: Argument --train or --infer not found")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent Method')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=False,
                        dest='train', help='Input --train to Train')
    group.add_argument('--infer', action='store_true',default=False,
                        dest='infer', help='Input --infer to Infer')
    parser.add_argument('--train_data_path', type=str,
                        default='feature/X_train', dest='train_data_path',
                        help='Path to training data')
    parser.add_argument('--train_label_path', type=str,
                        default='feature/Y_train', dest='train_label_path',
                        help='Path to training data\'s label')
    parser.add_argument('--test_data_path', type=str,
                        default='feature/X_test', dest='test_data_path',
                        help='Path to testing data')
    parser.add_argument('--save_dir', type=str,
                        default='logistic_params/', dest='save_dir',
                        help='Path to save the model parameters')
    parser.add_argument('--output_dir', type=str,
                        default='logistic_output/log_pred.csv', dest='output_dir',help='Path to save the model parameters')
    opts = parser.parse_args()
    main(opts)
