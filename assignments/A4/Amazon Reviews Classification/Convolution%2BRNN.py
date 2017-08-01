
# coding: utf-8

# In[1]:

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
from keras.layers import Input, Dense, Flatten, Lambda, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import SimpleRNN, GRU, LSTM, Bidirectional

import keras.backend as K

from Classifier import Classifier


# In[ ]:

class HybridClassifier(Classifier):
    """ Hybrid model, mainly Convolution + RNN (LSTM)
    """
    def __init__(self, batch_size, epochs, raw_data_path=None, embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100):
        super(HybridClassifier, self).__init__(batch_size, epochs, raw_data_path=None, embedded_data_path=embedded_data_path, embedding_dim=embedding_dim) 
        self.type = 'hybrid'

    def build(self):
        """ train a hybrid model with Convolution + LSTM
        """
        sequence_input = Input(shape=(self.max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        y = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        z = concatenate([x, y])
        preds = Dense(6, activation='softmax')(z)
        self.model = Model(sequence_input, preds)


# In[ ]:

if __name__ == "__main__":
    """ test the hybird classifier with Convolution + LSTM
    """
    
    conv_lstm = HybridClassifier(batch_size=128, epochs=20, raw_data_path=None, embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100)
    conv_lstm.build()
    conv_lstm.train()
    print("constructed Convolution + LSTM classifier")
    conv_lstm.evaluate()
    print("Convolution + LSTM classifier evaluated")

