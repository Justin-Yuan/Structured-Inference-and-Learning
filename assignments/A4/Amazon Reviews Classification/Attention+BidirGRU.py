
# coding: utf-8

# In[1]:

from __future__ import print_function 
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

import os
import sys
import time 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, Embedding, Activation, Reshape, merge, Merge
from keras.layers import SimpleRNN, GRU, LSTM, Bidirectional
from keras.engine.topology import Layer
from keras import initializers

import keras.backend as K

from Classifier import Classifier


# In[ ]:

class WordAttentionClassifier(Classifier):
    """ Text classifier with word-level attention 
    """
    def __init__(self, batch_size, epochs, raw_data_path=None, embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100):
        super(WordAttentionClassifier, self).__init__(batch_size, epochs, raw_data_path=None, embedded_data_path=embedded_data_path, embedding_dim=embedding_dim) 
        self.type = 'word_attention'

    def build(self):
        """ train a hybrid model with Convolution + LSTM
        """
        sentence_input = Input(shape=(self.max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sentence_input)
        gru_word = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
        dense_word = TimeDistributed((Dense(100)))(gru_word)
        tanh_word = TimeDistributed(Activation('tanh'))(dense_word)
        att_word = AttLayer()(tanh_word)
        preds = Dense(6, activation='softmax')(att_word)
        self.model = Model(sentence_input, preds)
        
    def train(self):
        """ use RMSprop instead of Adam 
        """
        optimizer='rmsprop'
        super(WordAttentionClassifier, self).train(optimizer=optimizer)
        
        
class AttLayer(Layer):
    """ Word-level attention layer 
    """
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(AttLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                     shape=(input_shape[-1], 1),
                                     initializer='normal',
                                     trainable=True)
        #print(self.kernel.get_shape())

        super(AttLayer, self).build(input_shape)
        
    def call(self, x, mask=None):
        eij = K.dot(x, self.kernel)
        
        ai = K.exp(eij)
        weights = ai/tf.expand_dims(K.sum(ai, axis=1), -1) #ai/K.sum(ai, axis=1).dimshuffle(0, 'x')
        
        weighted_input = x*weights #tf.expand_dims(weights, -1) #x*weights.dimshuffle(0, 1, 'x')
        return tf.reduce_sum(weighted_input, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# In[ ]:

if __name__ == "__main__":
    """ test the hybird classifier with Convolution + LSTM
    """
    
    attention_model = WordAttentionClassifier(batch_size=128, epochs=10, raw_data_path=None, embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100)
    attention_model.build()
    attention_model.train()
    print("constructed Word Attention classifier")
    attention_model.evaluate()
    print("Word Attention classifier evaluated")

