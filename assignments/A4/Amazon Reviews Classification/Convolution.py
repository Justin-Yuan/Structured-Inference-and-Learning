
# coding: utf-8

# In[5]:

from __future__ import print_function 
import numpy as np
import tensorflow as tf 

import os
import sys
import time 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import SimpleRNN, GRU, LSTM, Bidirectional

from keras.callbacks import ModelCheckpoint
import keras.backend as K

from Classifier import Classifier


# In[6]:

class ConvClassifier(Classifier):
    """
    """
    def __init__(self, batch_size, epochs, raw_data_path=None, embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100):
        super(ConvClassifier, self).__init__(batch_size, epochs, raw_data_path=None, embedded_data_path=embedded_data_path, embedding_dim=embedding_dim) 
        self.type = 'conv'

    def build(self):
        """ train a 1D convnet with global maxpooling
        """
        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(6, activation='softmax')(x)
        self.model = Model(sequence_input, preds)
        self.model.summary()

    def train(self, loss='categorical_crossentropy', optimizer='adam', model_base_path="models/"):
        """ for Simple RNN, the optimizer needs to implement gradients clipping to prevent explosion 
        """
        if self.type == 'simple':
            optimizer = optimizers.Adam(clipnorm=1.)
        model_base_path += self.type + '/'
        super(ConvClassifier, self).train(optimizer=optimizer, model_base_path=model_base_path)
    
    def save(self, path='models/model.h5'):
        """
        """
        path = 'models/'
        path += self.type + '/model.h5'
        super(ConvClassifier, self).save(path=path)

# In[3]:

if __name__ == "__main__":
    """ test the convolution classifier
    """
    
    conv_classifier = ConvClassifier(batch_size=128, epochs=10, raw_data_path=None, embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100)
    conv_classifier.build()
    conv_classifier.train()
    print("constructed convolution classifier")
    conv_classifier.evaluate()
    print("convolution classifier evaluated")

