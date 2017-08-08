
# coding: utf-8

# In[2]:

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
from keras.models import load_model


import keras.backend as K
from keras import optimizers

from Classifier import Classifier


# In[1]:

class RNNClassifier(Classifier):
    """
    """
    def __init__(self, batch_size, epochs, raw_data_path=None, embedded_data_path='../data/data_and_embedding100.npz', embedding_dim=100, rnn_type='lstm'):
        super(RNNClassifier, self).__init__(batch_size, epochs, raw_data_path=None, embedded_data_path=embedded_data_path, embedding_dim=embedding_dim) 
        
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
        sequence_input = Input(shape=(self.max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = SimpleRNN(50, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        preds = Dense(6, activation='softmax')(x)
        model_rnn_final_state = Model(sequence_input, preds)
        return model_rnn_final_state
    
    def build_lstm(self):
        """
        """
        sequence_input = Input(shape=(self.max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        preds = Dense(6, activation='softmax')(x)
        model_lstm_final_state = Model(sequence_input, preds)
        return model_lstm_final_state
    
    def build_gru(self):
        """
        """
        sequence_input = Input(shape=(self.max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = GRU(50, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        preds = Dense(6, activation='softmax')(x)
        model_gru_final_state = Model(sequence_input, preds)
        return model_gru_final_state
    
    def build_bidirectional_lstm(self):
        """
        """
        sequence_input = Input(shape=(self.max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2))(embedded_sequences)
        preds = Dense(6, activation='softmax')(x)
        model_bidirlstm_final_state = Model(sequence_input, preds)
        return model_bidirlstm_final_state

    def train(self, loss='categorical_crossentropy', optimizer='adam', model_base_path="../models/"):
        """ for Simple RNN, the optimizer needs to implement gradients clipping to prevent explosion 
        """
        if self.type == 'simple':
            optimizer = optimizers.Adam(clipnorm=1.)
        super(RNNClassifier, self).train(optimizer=optimizer)


# In[ ]:

if __name__ == "__main__":
    """ test different RNN models
    """
    
    # Simple RNN with gradient clipping
    simple_RNN = RNNClassifier(batch_size=128, epochs=30, raw_data_path=None, embedded_data_path='../data/data_and_embedding100.npz', embedding_dim=100, rnn_type='simple')
    if simple_RNN.model == None:
        simple_RNN.build()
        simple_RNN.train()
    print("constructed simple RNN classifier")
    simple_RNN.evaluate()
    print("simple RNN classifier evaluated")
    simple_RNN.save()
    print("simple RNN model saved")
    
    # LSTM 
    lstm = RNNClassifier(batch_size=128, epochs=30, raw_data_path=None, embedded_data_path='../data/data_and_embedding100.npz', embedding_dim=100, rnn_type='lstm')
    if lstm.model == None:
        lstm.build()
        lstm.train()
    print("constructed LSTM classifier")
    lstm.evaluate()
    print("LSTM classifier evaluated")
    lstm.save()
    print("LSTM model saved")
    
    # GRU
    gru = RNNClassifier(batch_size=128, epochs=30, raw_data_path=None, embedded_data_path='../data/data_and_embedding100.npz', embedding_dim=100, rnn_type='gru')
    if gru.model == None:
        gru.build()
        gru.train()
    print("constructed GRU classifier")
    gru.evaluate()
    print("GRU classifier evaluated")
    gru.save()
    print("GRU model saved")

    # Bidirectional LSTM
    bidirectional_lstm = RNNClassifier(batch_size=128, epochs=30, raw_data_path=None, embedded_data_path='../data/data_and_embedding100.npz', embedding_dim=100, rnn_type='bidirectional')
    if bidirectional_lstm.model == None:
        bidirectional_lstm.build()
        bidirectional_lstm.train()
    print("constructed bidirectional LSTM classifier")
    bidirectional_lstm.evaluate()
    print("bidirectional LSTM classifier evaluated")
    bidirectional_lstm.save()
    print("bidirectional LSTM model saved")

