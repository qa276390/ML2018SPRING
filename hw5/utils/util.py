
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import _pickle as pk
import gensim
print('gensim(3.1.0):',gensim.__version__)
import re

# In[2]:


class DataManager:
    def __init__(self):
        self.data = {}
        #self.emb_mat = {}
    # Read data from data_path
    #  name       : string, name of data
    #  with_label : bool, read data with label or without label
    def get_emb_mat(self):
        return self.emb_mat
    def get_lenofindex(self):
        return len(self.tokenizer.word_index)
    def add_emb_mat(self, emb_size):
        #emb_size = 300
        #data = self.data
        data = [ d.split(" ") for d in self.data['train_data'][0]]
        w2v_model = gensim.models.Word2Vec(data, size = emb_size, window = 10, min_count = 0, workers = 6, sg = 1)
        word_index = self.tokenizer.word_index

        mat = np.zeros((len(word_index), emb_size))
        oc = 0
        cc = 0
        I = 0
        for word, i in word_index.items():
            I = i
            try:
                embedding_vec = w2v_model.wv[word]
                mat[i] = embedding_vec
                cc += 1
            except:
                oc += 1
                #print(word)
        print('cc:',cc)
        print('oc:',oc)
        print('emb_mat shape:',np.shape(mat))
        self.emb_mat = mat
    
    def add_data(self,name, data_path, with_label=True):
        print ('read data from %s...'%data_path)
        X, Y = [], []
        with open(data_path,'r') as f:
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                else:
                    X.append(line.split(",", 1)[1])
        ##preprocessing data###
        stemmer = gensim.parsing.porter.PorterStemmer()
        print(X[:2])
        X_c = [self.preprocess(s) for s in X]
        X_c = [s for s in stemmer.stem_documents(X_c)]
        print(X_c[:2])
        #######################
        if with_label:
            self.data[name] = [X_c,Y]
        else:
            self.data[name] = [X_c]
        

    
        
    def preprocess(self, string):
        string = string.replace(" ' ", "")
        for same_char in re.findall(r'((\w)\2{2,})', string):
            string = string.replace(same_char[0], same_char[1])
        for digit in re.findall(r'\d+', string):
            string = string.replace(digit, "1")
        for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
            if punct[0:2] =="..":
                string = string.replace(punct, "...")
            else:
                string = string.replace(punct, punct[0])
        return string
    # Build dictionary
    #  vocab_size : maximum number of word in yout dictionary
    def tokenize(self, vocab_size):
        print ('create new tokenizer')
        #self.tokenizer = Tokenizer(num_words=vocab_size,filters="$%^")
        self.tokenizer = Tokenizer(num_words=vocab_size)
        print('num_words:',vocab_size)

        for key in self.data:
            print ('tokenizing %s'%key)
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)
        
    # Save tokenizer to specified path
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump(self.tokenizer, open(path, 'wb'))
            
    # Load tokenizer from specified path
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pk.load(open(path, 'rb'))
    
    # Convert words in data to index and pad to equal size
    #  maxlen : max length after padding
    def to_sequence(self, maxlen):
        self.maxlen = maxlen
        for key in self.data:
            print ('Converting %s to sequences'%key)
            tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen))
    
    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to tfidf'%key)
            self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='tfidf')
    
    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))
    
    def get_semi_data(self,name,label,threshold,loss_function) : 
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)
    # get data by name
    def get_data(self,name):
        return self.data[name]

    # split data to two part by a specified ratio
    #  name  : string, same as add_data
    #  ratio : float, ratio to split
    def split_data(self, name, ratio):
        data = self.data[name]
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])
    

