from __future__ import print_function 
import numpy as np
import tensorflow as tf 

import os
import sys
import time 
import pickle 

from sklearn import svm 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import keras.backend as K


class HierarchicalAttentionClassifier(object):
    """
    """
    def __init__(self, batch_size, epochs, raw_data_path=None, embedded_data_path=None,  embedding_dim=None):
        """
        """
        self.batch_size = batch_size
        self.epochs = epochs

        # data placeholders
        self.x_train = None 
        self.x_val = None 
        self.x_test = None 
        self.y_train = None 
        self.y_val = None 
        self.y_test = None 

        # model type 
        self.type = 'hier_attention'
        
        # # load data selectively 
        # if raw_data_path != None:
        #     self._load_raw_data(raw_data_path)
        # if embedded_data_path != None:
        #     self._load_embedded_data(embedded_data_path)
            
        # variable to hold the model 
        self.model = None 
        try:
            self.load()
            print("Loaded existing model from model.h5")
        except:
            print("No existing model yet")

        # construct an embedding layer (only necessary for logistic regression)
        if embedding_dim != None:
            self.embedding_dim = embedding_dim 
            self.embedding_layer = self._construct_embedding_layer()
    
    def _construct_embedding_layer(self):
        """
        """
        return Embedding(self.num_words,
                        self.embedding_dim,
                        weights=[self.embedding_matrix],
                        input_length=self.max_sequence_length,
                        trainable=False)
        
    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        
    def set_epochs(self, new_epochs):
        self.epochs = new_epochs
        
    # def _load_raw_data(self, raw_data_path):
    #     """ saved data format 
    #     processed_data ={
    #         'texts': filtered_texts,
    #         'scores': scores,
    #         'scores_dict':scores_dict,
    #         'count': count, 
    #         'embeddings_index': embeddings_index,
    #         'word_index':word_index
    # }
    #     """
    #     with open(raw_data_path, 'rb') as f:
    #         raw_data = pickle.load(f)
    #         self.texts = raw_data['texts']
    #         self.scores = raw_data['scores']
    #         self.scores_dict = raw_data['scores_dict']
    #         print(self.scores_dict)
    #         self.count = raw_data['count'] 
    #         self.word_index = raw_data['word_index']
    #         print('loaded raw processed data from', raw_data_path)
        
    # def _load_embedded_data(self, embedded_data_path, validation_split=0.1):
    #     """Classifier
    #     """ 
    #     # f = np.load('data_and_embedding100.npz')
    #     f = np.load(embedded_data_path)
        
    #     self.num_labels = int(f['num_labels']) + 1
    #     self.num_words = int(f['num_words'])
    #     self.embedding_dim = int(f['embedding_dim'])
    #     self.max_sequence_length = int(f['max_sequence_length'])

    #     self.x = f['x_train']
    #     self.y = f['y_train']
    #     self.x_test = f['x_test']
    #     self.y_test = f['y_test']

    #     self.embedding_matrix = f['embedding_matrix']
        
    #     indices = np.arange(self.x.shape[0])
    #     np.random.shuffle(indices)
    #     self.x = self.x[indices]
    #     self.y = self.y[indices]
    #     num_validation_samples = int(validation_split * self.x.shape[0])

    #     self.x_train = self.x[:-num_validation_samples]
    #     self.y_train = self.y[:-num_validation_samples]
    #     self.x_val = self.x[-num_validation_samples:]
    #     self.y_val = self.y[-num_validation_samples:]
    #     print('loaded embedded datasets from', embedded_data_path)



    def build(self):
        """ train a hybrid model with Convolution + LSTM
        """
        sentence_input = Input(shape=(max_sequence_length, ), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        gru_word = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
        dense_word = TimeDistributed((Dense(100)))(gru_word)
        tanh_word = TimeDistributed(Activation('tanh'))(dense_word)
        att_word = AttLayer()(tanh_word)
        sentEncoder = Model(sentence_input, att_word)

        review_input = Input(shape=(max_sentence_length, max_sequence_length), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        gru_sent = Bidirectional(GRU(50, return_sequences=True))(review_encoder)
        dense_sent = TimeDistributed(Dense(100))(gru_sent)
        tanh_sent = TimeDistributed(Activation('tanh'))(dense_sent)
        att_sent = AttLayer()(tanh_sent)
        preds = Dense(6, activation='softmax')(att_sent)
        model_attention = Model(review_input, preds)
        self.model = model_attention
        

    def train(self, loss='categorical_crossentropy', optimizer='rmsprop', model_base_path="models/"):
        """ 
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
        
        if self.type != None:
            model_base_path += self.type + '/'
        filepath= model_base_path + "weights-improvement-{epoch:02d}-{acc:.2f}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        
        start_time = time.time()
        self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(self.x_val, self.y_val),
                    callbacks=callbacks_list)

        print("Training time: ", time.time() - start_time)
        

    def predict(self, test_data):
        """ predict lables given the test data 
        """
        predictions = self.model.predict(test_data)
        return predictions
        

    def evaluate(self, x_test_data=None, y_test_data=None):
        """ evaluate on the test set or specified test data 
        """
        if x_test_data == None or y_test_data == None:
            res = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        else:
            res = self.model.evaluate(x_test_data, y_test_data, verbose=0)
        print("accuracy is %.2f" % (res[1]*100), end='')  # model.metrics_names[1] is acc
        print('%')

    def summary(self):
        """ get model summary if it exists
        """
        try:
            self.model.summary()
        except:
            print("model summary is not available")
    
    def save(self, path='models/model.h5'):
        """ save the model to default location 
        """
        if self.type != None:
            path = 'models/'
            path += self.type + '/model.h5'
        self.model.save(path)

    def load(self, path='models/model'):
        """ load the default model 
        """
        if self.type != None:
            path = 'models/'
            path += self.type + '/model.h5'
        self.model = load_model(path)
        print("model loaded")
    

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




if __name__ == '__main__':
    """ test the Hierarchical attention classifier
    """


