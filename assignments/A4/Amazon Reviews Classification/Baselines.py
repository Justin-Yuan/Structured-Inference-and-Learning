
# coding: utf-8

# ## Baselines

# In[2]:

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

import keras.backend as K

from Classifier import Classifier

# In[4]:

class MajorityClass(Classifier):
    """ the self.model variable holds the majority class 
    """
    def __init__(self, batch_size, epochs, raw_data_path='data/raw_processed_data.pkl', embedded_data_path='data/data_and_embedding100.npz'):
        super(MajorityClass, self).__init__(batch_size, epochs, raw_data_path=raw_data_path, embedded_data_path=embedded_data_path)
    
    def build_majority_predictor(self):
        """ construct the label distribution dict in training set,
            return the majority class as baseline predictor 
        """
        max_occr = max(list(self.scores_dict.values()))
        for label in self.scores_dict:
            if self.scores_dict[label] == max_occr:
                self.model = label
                
        for key in self.scores_dict:
            print('class', key, ':', self.scores_dict[key]/self.count)
            
    def predict_majority_predictor(self, test_data):
        """ for the majority predictor, the model itself is the majority label 
        """
        predictions = self.model * np.ones(test_data.shape[0]) 
        return predictions 
    
    def evaluate_majority_predictor(self, y_test_data=None):
        """
        """
        print("The majority class is", self.model)
        if y_test_data == None:
            preds = np.zeros(shape=(self.y_test.shape[0], self.y_test.shape[1]))
            preds[:, self.model] = 1 
            acc = np.mean(1*np.equal(np.array(self.y_test), preds))
        else:
            pred = self.model * np.ones(shape=(y_test_data.shape[0], y_test_data.shape[1]))
            acc = np.mean(1*np.equal(np.array(self.y_test_data), preds))
        print("accuracy is", acc)
    
    def save_model(self):
        """ save the trained model to the 'models/' directory
        """
        with open('models/majority_class.pkl', 'wb') as f:
            pickle.dump(self.model, f)
    


# In[5]:

class LogisticRegression(Classifier):
    """
    """
    def __init__(self, batch_size, epochs, raw_data_path=None, embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100):
        super(LogisticRegression, self).__init__(batch_size, epochs, raw_data_path=None, embedded_data_path=embedded_data_path, embedding_dim=embedding_dim) 

    def build_logistic_regression(self):
        """
        """
        sequence_input = Input(shape=(self.max_sequence_length, ), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = Lambda(self.embedding_mean)(embedded_sequences)
        preds = Dense(self.num_labels, activation='softmax')(x)

        model = Model(sequence_input, preds)
        #model.summary()
        self.model = model
    
    def train_logistic_regression(self, loss='categorical_crossentropy', optimizer='adam', ):
        """
        """
        self.model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['acc']) 
        
        start_time = time.time()

        self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(self.x_val, self.y_val), verbose=0)

        print("Training time: ", time.time() - start_time)
        
    def predict_logistic_regression(self, test_data):
        """ feed into the logistic regression model to get predictions 
        """
        predictions = self.model.predict(test_data)
        return predictions
    
    def evaluate_logistic_regression(self, x_test_data=None, y_test_data=None):
        """
        """
        if x_test_data == None or y_test_data == None:
            res = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        else:
            res = self.model.evaluate(x_test_data, y_test_data, verbose=0)
        print("accuracy is %.2f" % (res[1]*100), end='')  # model.metrics_names[1] is acc
        print('%')
    
    def embedding_mean(self, x):
        """ for logistic regression model 
        """
        return tf.reduce_mean(x, axis=1)
    
    def save_model(self):
        """ save the trained model to the 'models/' directory
        """
        self.model.save('models/logistic_regression.h5')


# In[6]:

class SVM(Classifier):
    """
    """
    def __init__(self, batch_size, epochs=50000, raw_data_path=None, 
                 embedded_data_path='data/data_and_embedding100.npz', model_type='embedding'):
        """
        """
        super(SVM, self).__init__(batch_size, epochs=epochs, raw_data_path=None, embedded_data_path=embedded_data_path)
        self.model_type = model_type   # bow (bag of words) or embedding
        
        # load or construct dataset for SVM
        self._construct_SVM_data()
        
    def _construct_SVM_data(self):
        """ load the saved data or construct a new one 
        """
        try:
            if self.model_type == 'bow':
                f = np.load("data/svm_bow_data")
                self.x_train_bow = f['x_train_bow']
                self.x_val_bow = f['x_val_bow']
                self.x_test_bow = f['x_test_bow']
            elif self.model_type == 'embedding':
                f = np.load("data/svm_embedding_data")
                self.x_train_embedded = f['x_train_embedded']
                self.x_val_embedded = f['x_val_embedded']
                self.x_test_embedded = f['x_test_embedded']
                
            self.y_train_svm = f['y_train_svm']
            self.y_val_svm = f['y_val_svm']
            self.y_test_svm = f['y_test_svm']
        except:     
            if self.model_type == 'bow':
                self.x_train_bow = self.convert_doc_feature_vec(self.x_train, self.embedding_matrix)
                self.x_val_bow = self.convert_doc_feature_vec(self.x_val, self.embedding_matrix)
                self.x_test_bow = self.embed_doc(self.x_test, self.embedding_matrix)
            elif self.model_type == 'embedding':
                self.x_train_embedded = self.embed_doc(self.x_train, self.embedding_matrix)
                self.x_val_embedded = self.embed_doc(self.x_val, self.embedding_matrix)
                self.x_test_embedded = self.embed_doc(self.x_test, self.embedding_matrix)

            self.y_train_svm = self.convert_labels(self.y_train)
            self.y_val_svm = self.convert_labels(self.y_val)
            self.y_test_svm = self.convert_labels(self.y_test)
            
            self._save_SVM_data()
        
    def _save_SVM_data(self):
        """
        """
        if self.model_type == 'bow':
            np.savez("data/svm_bow_data",
                x_train_bow = self.x_train_bow,
                x_val_bow = self.x_val_bow,
                x_test_bow = self.x_test_bow,
                y_train_svm = self.y_train_svm,
                y_val_svm = self.y_val_svm,
                y_test_svm = self.y_test.svm)
        elif self.model_type == 'embedding':
            np.savez("data/svm_embedding_data",
                x_train_embedded = self.x_train_embedded,
                x_val_embedded = self.x_val_embedded,
                x_test_embedded = self.x_test_embedded,
                y_train_svm = self.y_train_svm,
                y_val_svm = self.y_val_svm,
                y_test_svm = self.y_test_svm)
        
    def build_SVM(self):
        """
        """
        self.model = svm.LinearSVC(max_iter=self.epochs, verbose=1)
        
    def train_SVM(self):
        """
        """
        if self.model_type == 'bow':
            self.model.fit(self.x_train_bow, self.y_train_svm)
        elif self.model_type == 'embedding':
            self.model.fit(self.x_train_embedded, self.y_train_svm)
    
    def predict_SVM(self, x_test_data):
        """
        """
        preds = self.model.predict(x_test_data)
        return preds 
        
    def evaluate_SVM(self):
        """
        """
        if self.model_type == 'bow':
            preds = self.model.predict(self.x_test_bow)
            acc = np.mean(1*np.equal(np.array(self.y_test_svm), preds))
        elif self.model_type == 'embedding':
            preds = self.model.predict(self.x_test_embedded)
            acc = np.mean(1*np.equal(np.array(self.y_test_svm, dtype=preds.dtype), preds))
        print("accuracy: %g" % (acc*100), end='')
        print("%")

    # Bag of words 
    # implementation is flawed, consuming too much memory 
    def construct_feature_vec(self, text, embedding_matrix):
        text_vec = [0] * embedding_matrix.shape[0]
        zero_flag = 1
        for word in text:
            if zero_flag and word < 1:
                continue 
            else:
                zero_flag = 0
                text_vec[word] += 1
        return text_vec 

    def convert_doc_feature_vec(self, doc, embedding_matrix):
        return [self.construct_feature_vec(text, embedding_matrix) for text in doc]
    
    # Word embedding 
    def embed_text(self, text, embedding_matrix):
        instance_count = 0
        text_embedding = np.zeros(embedding_matrix[0].shape)
        for word in text:
            if word != 0:
                instance_count += 1
                text_embedding +=  embedding_matrix[word]
        return text_embedding/instance_count 

    def embed_doc(self, doc, embedding_matrix):
        return [self.embed_text(text, embedding_matrix) for text in doc]

    def convert_labels(self, one_hot_labels):
        return [list(label).index(1.0) for label in one_hot_labels]
    


# ### main code

# In[86]:

if __name__ == '__main__':
    """ test the baseline classifiers 
    """
    
    majority_classifier = MajorityClass(batch_size=128, epochs=10, raw_data_path='data/raw_processed_data.pkl')
    majority_classifier.build_majority_predictor()
    print('constructed majority class classifier')
    majority_classifier.evaluate_majority_predictor()
    print("majority class preditor evaluated", end='\n\n')
    
    logistic_classifier = LogisticRegression(batch_size=128, epochs=10, raw_data_path=None, embedded_data_path='data/data_and_embedding100.npz', embedding_dim=100)
    logistic_classifier.build_logistic_regression()
    logistic_classifier.train_logistic_regression()
    print('constructed logitic regression classifier')
    logistic_classifier.evaluate_logistic_regression()
    print("logistic regression evaluated")
    
    svm_classifier = SVM(batch_size=128, epochs=100000, raw_data_path=None, 
                 embedded_data_path='data/data_and_embedding100.npz', model_type='embedding')
    svm_classifier.build_SVM()
    svm_classifier.train_SVM()
    print('constructed SVM classifier')
    svm_classifier.evaluate_SVM()
    print("SVM with embedding evaluated")
    
    

