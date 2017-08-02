
# coding: utf-8

"""
Justin Yuan, Aug 1st, 2017

Part of speech tagging dataset reference: http://www.cnts.ua.ac.be/conll2000/chunking/
"""


import numpy as np
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D, Bidirectional

from keras.callbacks import ModelCheckpoint, EarlyStopping


# Map to indices / encode texts and tagging labels 

def encode_one_hot(idx, dim):
    """ use one-hot encoding for the labels
    """
    temp = [0]*dim
    temp[idx] = 1
    return temp

def encode_corpus(X, maxlen):
    """ map words into indices 
    """
    X_enc = [[word2ind[word] for word in x] for x in X]
    return pad_sequences(X_enc, maxlen=maxlen, value=0)

def encode_labels(Y, maxlen, dim):
    """ map labels into one-hot vectors 
    """
    Y_enc = [[label2ind[tag] for tag in y] for y in Y]
    Y_enc = pad_sequences(Y_enc, maxlen=maxlen, value=0)
    Y_enc = [[encode_one_hot(idx, dim) for idx in y] for y in Y_enc]
    return np.array(Y_enc)


# Split the dataset into training and validation sets

def split_training_validation(X_train_enc, y_train_enc, ratio=0.1):
    """ split the training data further into a training set and a validation set 
    """
    validation_split = ratio

    X_enc = X_train_enc
    y_enc = y_train_enc

    indices = np.arange(X_enc.shape[0])
    np.random.shuffle(indices)
    X_enc = X_enc[indices]
    y_enc = y_enc[indices]
    num_validation_samples = int(validation_split * X_enc.shape[0])

    X_train_enc = X_enc[:-num_validation_samples]
    y_train_enc = y_enc[:-num_validation_samples]
    X_val_enc = X_enc[-num_validation_samples:]
    y_val_enc = y_enc[-num_validation_samples:]
    return X_train_enc, y_train_enc, X_val_enc, y_val_enc


# Evaluate the model 

def unpad_sequences(yh, pr):
    """ remove the padding 0s for the ground truth tags and predicted tags 
    """
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    return yh, ypr

def score(yh, pr):
    """ flatten tags in lsit of samples into a list of tags 
    """
    yh, ypr = unpad_sequences(yh, pr)
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

def compare_prediction_groundtruth(model, X, y, verbose=True, indices=None):
    """ show evaluation results, including prediction accuracy (word-wise) and the confusion matrix, 
        optionally showing the predicted tags and groundtruth tags for a chosen set of samples (a list of indices as an argument)
    """
    pr = model.predict(X) 
    pr = pr.argmax(2)
    yh = y.argmax(2)
    fyh, fpr = score(yh, pr)
    # get accuracy score 
    acc = accuracy_score(fyh, fpr)
    # get confusion matrix
    cm = confusion_matrix(fyh, fpr)
    print('Accuracy:', acc, end='\n\n')
    print('Confusion matrix:')
    print(cm, end='\n\n')
    
    if verbose and indices != None:
        yh, ypr = unpad_sequences(yh, pr)
        for idx in indices:
            print('test sample', idx)
            print([ind2label[index] for index in yh[idx]])
            print([ind2label[index] for index in ypr[idx]], end='\n\n')
    return acc, cm

def get_TP_FP_FN(cm, label):
    """ get numbers of True positives, False positives and False negitives,
        cm is the confusion matrix for multiple labels, label is a tag index
    """
    dim = min(cm.shape[0], cm.shape[1])
    if label >= 0 and label < dim:
        # TP, True positive, diagonal position
        tp = cm[label, label]
        # FP, False positive: sum of column label (without main diagonal)
        fp = sum(cm[:, label]) - cm[label][label]
        # FN, False negative: sum of row label (without main diagonal)
        fn = sum(cm[label, :]) - cm[label][label]
        return tp, fp, fn
    else:
        print("label out of bound")

def get_precision(cm, label):
    # precision = TP / (TP + FP)
    tp, fp, fn = get_TP_FP_FN(cm, label)
    return tp / (tp + fp)

def get_recall(cm, label):
    # recall = TP / (TP + FN)
    tp, fp, fn = get_TP_FP_FN(cm, label)
    return tp / (tp + fn)

def get_F1_score(cm, label):
    # F1 = 2TP / (2TP + FP + FN)
    tp, fp, fn = get_TP_FP_FN(cm, label)
    return 2*tp / (2*tp + fp + fn)
        
def get_evaluation_statistics(ind2label, label=0):
    """ show True positives, False positives, False negatives, precision, recall and F1 score for a particular label
    """
    print("evaluation statistics for label", label, ind2label[label])
    tp, fp, fn = get_TP_FP_FN(cm, label)
    print("True positives", tp, " , False positives", fp, " , False negatives", fn)
    precison = get_precision(cm, label)
    print("Precision", precision)
    recall = get_recall(cm, label)
    print("Recall", recall)
    f1 = get_F1_score(cm, label)
    print("F1 score", f1, end='\n\n')
    


if __name__ == "__main__":
    """ train and test POS tagging on the Conll dataset 
    """

    # load the datasets 

    with open('data/pos_conll.pkl', 'rb') as f:
        data = pickle.load(f)

    X_train = data['train']['X'] 
    tags_train = data['train']['tags'] 
    chunks_train = data['train']['chunks'] 

    X_test = data['test']['X'] 
    tags_test = data['test']['tags'] 
    chunks_test = data['test']['chunks'] 

    maxlen = data['stats']['maxlen']
    word2ind = data['stats']['word2ind']
    ind2word = data['stats']['ind2word'] 
    label2ind = data['stats']['label2ind'] 
    ind2label = data['stats']['ind2label'] 


    # Map to indices / encode texts and tagging labels 

    X_train_enc = encode_corpus(X_train, maxlen)
    y_train_enc = encode_labels(tags_train, maxlen, dim)


    # Split the dataset into training and validation sets
    
    X_train_enc, y_train_enc, X_val_enc, y_val_enc = split_training_validation(X_train_enc, y_train_enc, ratio=0.1)

    # POS Tagging Model 
    # model hyperparameters 
    max_features = len(word2ind) + 1    # plus one for the padding 0 token 
    embedding_size = 100
    hidden_size = 32
    out_size = len(label2ind) + 1   # plus one for the padding 0 token 
    batch_size = 32
    epochs = 30

    # Define the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_size,
                        input_length=maxlen, mask_zero=False))              # embed words with Keras default, mask_zero is False due to the need for convolution layer 
    model.add(Convolution1D(64,5,padding='same', activation='relu'))        # convolution layer captures local information 
    model.add(Dropout(0.25))
    model.add(Bidirectional(GRU(50, return_sequences=True)))                # bidirectional GRU to accumulated information in both forward and backward manners
    model.add(TimeDistributed(Dense(out_size, activation='softmax')))       # a dense layer for each hidden state to classify POS tag  

    model.summary()

    # Training 
    model.compile('rmsprop', 'categorical_crossentropy')

    # set checkpoints and early stopping for model training 
    filepath = "models/POS-conll-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
    callbacks_list = [checkpoint, earlystopping]

    # model training, with validation and callback functions (checkpoint the best model based on validatoin loss 
    #                                                         and apply early stopping if no improvement is shown in 8 consecutive epochs)
    model.fit(X_train_enc, y_train_enc, batch_size=batch_size, epochs=epochs,        # TN
        tn = 
            validation_data=(X_val_enc, y_val_enc), callbacks=callbacks_list)

    model.save('models/POS_Conll.h5')


    # evaluate the model 
    model = load_model('models/POS_Conll.h5')

    # constructing test data 
    X_test_enc = encode_corpus(X_test, maxlen)
    y_test_enc = encode_labels(tags_test, maxlen, dim)categorical_crossentropy

    # return the test set score(categorical_crossentropy / loss), NOT USEFUL  
    score = model.evaluate(X_test_enc, y_test_enc, batch_size=batch_size, verbose=0)
    print('Raw test score:', score)    

    # better evaluation, shows accuracy of the model on a word level and the confusion matrix of the test set 
    acc, cm = compare_prediction_groundtruth(model, X_test_enc, y_test_enc, True, indices=[1,2,3])

    # show evaluation statistics of chosen labels
    labels = [1, 2, 3]
    for label in labels:
        get_evaluation_statistics(ind2label, label)
