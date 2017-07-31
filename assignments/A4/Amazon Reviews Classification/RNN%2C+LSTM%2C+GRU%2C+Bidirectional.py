
# coding: utf-8

# In[2]:

from __future__ import print_function 
import numpy as np
import tensorflow as tf 

import os
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import SimpleRNN, GRU, LSTM, Bidirectional

import keras.backend as K
from keras import optimizers


# In[1]:

class RNNClassifier():
    """
    """
    def __init__(self, batch_size, epochs, raw_data_path=None, embedded_data_path='data/data_and_embedding100.npz',
                 embedding_dim=100, rnn_type='lstm'):
        super(RNNClassifier, self).__init__(batch_size, epochs, raw_data_path=None,
                embedded_data_path=embedded_data_path, embedding_dim=embedding_dim) 
        
        # get the RNN model type 
        self.type = rnn_type
        
    def build(self):
        """ 
        """
        if self.type == 'simple':
            self.model = self.build_simple_rnn()
        elif self.type == 'lstm':
            self.model = self.build_lstm()
        elif self.type == 'gru':
            self.model = self.build_gru()
        elif self.type == 'bidirectional':
            self.model = self.build_bidirectional_lstm()
            
    def build_simple_rnn(self):
        """
        """
        sequence_input = Input(shape=(max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = SimpleRNN(50, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        preds = Dense(6, activation='softmax')(x)
        model_rnn_final_state = Model(sequence_input, preds)
        return model_rnn_final_state
    
    def build_lstm(self):
        """
        """
        sequence_input = Input(shape=(max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        preds = Dense(6, activation='softmax')(x)
        model_lstm_final_state = Model(sequence_input, preds)
        print(model_lstm_final_state)
    
    def build_gru(self):
        """
        """
        sequence_input = Input(shape=(max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = GRU(50, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        preds = Dense(6, activation='softmax')(x)
        model_gru_final_state = Model(sequence_input, preds)
        return model_gru_final_state
    
    def build_bidirectional_lstm(self):
        """
        """
        sequence_input = Input(shape=(max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2))(embedded_sequences)
        preds = Dense(6, activation='softmax')(x)
        model_bidirlstm_final_state = Model(sequence_input, preds)
        return model_bidirlstm_final_state
        
    def train(self, loss='categorical_crossentropy', optimizer='adam', model_base_path="models/"):
        """ for Simple RNN, the optimizer needs to implement gradients clipping to prevent explosion 
        """
        if self.type == 'simple':
            optimizer = optimizers.Adam(clipnorm=1.)
        model_base_path += self.type + '/'
        super(RNNClassifier, self).train(optimizer=optimizer, model_base_path=model_base_path)
    
    def save(self, path='models/model.h5'):
        """
        """
        path = 'models/'
        path += self.type + '/model.h5'
        super(RNNClassifier, self).save(path=path)


# In[ ]:

if __name__ == "__main__":
    """ test different RNN models
    """
    
    # Simple RNN with gradient clipping
    simple_RNN = RNNClassifier(batch_size=128, epochs=10, raw_data_path=None,
                        embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100, rnn_type='simple')
    simple_RNN.build()
    simple_RNN.train()
    print("constructed simple RNN classifier")
    simple_RNN.evaluate()
    print("simple RNN classifier evaluated")
    
    # LSTM 
    lstm = RNNClassifier(batch_size=128, epochs=10, raw_data_path=None,
                        embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100, rnn_type='lstm')
    lstm.build()
    lstm.train()
    print("constructed LSTM classifier")
    lstm.evaluate()
    print("LSTM classifier evaluated")
    
    # GRU
    gru = RNNClassifier(batch_size=128, epochs=10, raw_data_path=None,
                        embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100, rnn_type='gru')
    gru.build()
    gru.train()
    print("constructed GRU classifier")
    gru.evaluate()
    print("GRU classifier evaluated")
    
    # Bidirectional LSTM
    bidirectional_lstm = RNNClassifier(batch_size=128, epochs=10, raw_data_path=None,
                        embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100, rnn_type='bidirectional')
    bidirectional_lstm.build()
    bidirectional_lstm.train()
    print("constructed bidirectional LSTM classifier")
    bidirectional_lstm.evaluate()
    print("bidirectional LSTM classifier evaluated")

