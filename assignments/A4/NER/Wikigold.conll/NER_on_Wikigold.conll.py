
""" 
Justin Yuan, Aug 1st, 2017

the model construction and evaluation procedures are very similar to the POS example, 
the program is mostly written in a sequential way for convenience with some utility functions (mainly for evaluation as in POS)

Reference: https://gist.github.com/dirko/1d596ca757a541da96ac3caa6f291229
"""
# coding: utf-8

import pickle 
import numpy as np 

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from keras.layers import Input, Dense, TimeDistributed
from keras.layers import Embedding, Activation
from keras.layers import GRU, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.backend import tf


def encode(x, n):
    """ apply one-hot encoding
    """
    result = np.zeros(n)
    result[x] = 1
    return result

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

def compare_prediction_groundtruth(model, X, y, ind2label, verbose=True, indices=None):
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
    precision = get_precision(cm, label)
    print("Precision", precision)
    recall = get_recall(cm, label)
    print("Recall", recall)
    f1 = get_F1_score(cm, label)
    print("F1 score", f1, end='\n\n')




if __name__ == "__main__":
    """ NER on Wikigold.conll dataset
    """

    # Load the data

    with open('data/conll.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['train']['X']
    y = data['train']['y']
    X_test = data['test']['X']
    y_test = data['test']['y']
    word2ind = data['stats']['word2ind']
    ind2word = data['stats']['ind2word']
    label2ind = data['stats']['label2ind']
    ind2label = data['stats']['ind2label']

    maxlen = max([len(x) for x in X])
    print('Maximum sequence length:', maxlen)

    X_enc = [[word2ind[c] for c in x] for x in X]
    X_enc = pad_sequences(X_enc, maxlen=maxlen)

    max_label = max(label2ind.values()) + 1

    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
    y_enc = pad_sequences(y_enc, maxlen=maxlen)


    # construct the test data 

    X_test_enc = [[word2ind[c] for c in x] for x in X_test]
    X_test = pad_sequences(X_test_enc, maxlen=maxlen)

    y_test_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_test]
    y_test_enc = [[encode(c, max_label) for c in ey] for ey in y_test_enc]
    y_test = pad_sequences(y_test_enc, maxlen=maxlen)


    # split the current training set into a training set and a validation set 

    validation_split = 0.1

    indices = np.arange(X_enc.shape[0])
    np.random.shuffle(indices)
    X_enc = X_enc[indices]
    y_enc = y_enc[indices]
    num_validation_samples = int(validation_split * X_enc.shape[0])

    X_train = X_enc[:-num_validation_samples]
    y_train = y_enc[:-num_validation_samples]
    X_val = X_enc[-num_validation_samples:]
    y_val = y_enc[-num_validation_samples:]
        
    print("sample distributions: ")
    print("# of training:", X_train.shape[0], ", # of validation:", X_val.shape[0], ", # of test", X_test.shape[0])

    print('training input and output shapes:')
    print(X_train.shape, y_train.shape)


    # Build the model 
    # model hyperparameters
    max_features = len(word2ind)+1
    embedding_size = 128
    hidden_size = 32
    out_size = len(label2ind) + 1
    batch_size = 32
    epochs = 10

    # model architecture
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_size,
                        input_length=maxlen, mask_zero=True))                   # embedding layer, with default zero padding
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))          # bidirecitonal LSTM
    model.add(TimeDistributed(Dense(out_size)))                                 # apply a dense layer on each hidden states  
    model.add(Activation('softmax'))

    model.summary()


    # Train the model 

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    filepath = "models/NER-Wikigold-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    callbacks_list = [checkpoint, earlystopping]
    
    # to get NER classifications

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=(X_val, y_val), callbacks=callbacks_list)
    model.save('models/NER_Wikigold.h5')


    #  Evaluate the model

    model = load_model('models/NER_Wikigold.h5')

    # get the test set score (categorical crossentropy / loss), not USEFUL 
    test_score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print('Raw test score:', test_score)

    # shows accuracy of the model on a word level and the confusion matrix of the test set 
    acc, cm = compare_prediction_groundtruth(model, X_test, y_test, ind2label, True, indices=[1,2,3])

    # show evaluation statistics of chosen labels
    labels = [1, 2, 3]
    for label in labels:
        get_evaluation_statistics(ind2label, label)
