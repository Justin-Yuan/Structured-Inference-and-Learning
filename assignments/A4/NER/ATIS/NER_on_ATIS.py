"""
Justin Yuan, Aug 1st, 2017

NER on the ATIS dataset, references (code and data sources) are in the References folder
Model construction and evaluation are almost identical to NER on Wikigold.conll, except using the larger ATIS dataset
"""
# coding: utf-8

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


def encode_one_hot(idx, dim):
    temp = [0]*dim
    temp[idx] = 1
    return temp

def encode_corpus(X, maxlen):
    #X_enc = [[w2idx[word] for word in x] for x in X]
    return pad_sequences(X, maxlen=maxlen, value=vocab_len)

def encode_labels(Y, maxlen, dim):
    #Y_enc = [[labels2idx[tag] for tag in y] for y in Y]
    Y_enc = pad_sequences(Y, maxlen=maxlen, value=dim-1)
    Y_enc = [[encode_one_hot(idx, dim) for idx in y] for y in Y_enc]
    return np.array(Y_enc)

def unpad_sequences(yh, pr, bound):
    """ remove the padding 0s for the ground truth tags and predicted tags 
    """
    coords = [np.where(yhh < bound)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    return yh, ypr

def score(yh, pr, bound):
    """ flatten tags in lsit of samples into a list of tags 
    """
    yh, ypr = unpad_sequences(yh, pr, bound)
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

def compare_prediction_groundtruth(model, X, y, idx2la, verbose=True, indices=None):
    """ show evaluation results, including prediction accuracy (word-wise) and the confusion matrix, 
        optionally showing the predicted tags and groundtruth tags for a chosen set of samples (a list of indices as an argument)
    """
    pr = model.predict(X) 
    pr = pr.argmax(2)
    yh = y.argmax(2)

    bound = len(idx2la) - 1
    fyh, fpr = score(yh, pr, bound)

    # get accuracy score 
    acc = accuracy_score(fyh, fpr)
    # get confusion matrix
    cm = confusion_matrix(fyh, fpr)
    print('Accuracy:', acc, end='\n\n')
    print('Confusion matrix:')
    print(cm, end='\n\n')
    
    if verbose and indices != None:
        yh, ypr = unpad_sequences(yh, pr, bound)
        for idx in indices:
            print('test sample', idx)
            print([idx2la[index] for index in yh[idx]])
            print([idx2la[index] for index in ypr[idx]], end='\n\n')
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
    """ NER on ATIS
    """

    # Load the data

    with open('data/atis.pkl', 'rb') as f:
        train_set, valid_set, test_set, dicts = pickle.load(f)

    w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

    # Create index to word/label dicts
    idx2w  = {w2idx[k]:k for k in w2idx}
    idx2ne = {ne2idx[k]:k for k in ne2idx}
    idx2la = {labels2idx[k]:k for k in labels2idx}
    
    # add one more label for padding in the index-word mapping dictionary 
    dim = len(idx2la) + 1
    idx2la[dim-1] = 'Padding' 

    vocab_len = len(w2idx)

    train_x, train_ne, train_label = train_set
    val_x, val_ne, val_label = valid_set
    test_x, test_ne, test_label = test_set

    X_full = train_x + val_x + test_x
    maxlen = max([len(x) for x in X_full])
    print('Maximum sequence length:', maxlen)


    X_enc = encode_corpus(train_x + val_x, maxlen)
    y_enc = encode_labels(train_label + val_label, maxlen, dim)

    X_test_enc = encode_corpus(test_x, maxlen)
    y_test_enc = encode_labels(test_label, maxlen, dim)


    # split the current training set into a training set and a validation set  

    validation_split = 0.1

    indices = np.arange(X_enc.shape[0])
    np.random.shuffle(indices)
    X_enc = X_enc[indices]
    y_enc = y_enc[indices]
    num_validation_samples = int(validation_split * X_enc.shape[0])

    X_train_enc = X_enc[:-num_validation_samples]
    y_train_enc = y_enc[:-num_validation_samples]
    X_val_enc = X_enc[-num_validation_samples:]
    y_val_enc = y_enc[-num_validation_samples:]

    print("sample distributions: ")
    print("# of training:", X_train_enc.shape[0], ", # of validation:", X_val_enc.shape[0], ", # of test", X_test_enc.shape[0])

    print('training input and output shapes:')
    print(X_train_enc.shape, y_train_enc.shape)


    # Build the model

    max_features = len(w2idx)+1
    embedding_size = 100
    hidden_size = 32
    out_size = len(labels2idx) + 1
    batch_size = 32
    epochs = 10


    # Define model
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_size,
                        input_length=maxlen, mask_zero=False))
    model.add(Convolution1D(64,5,padding='same', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Bidirectional(GRU(hidden_size, return_sequences=True)))
    model.add(TimeDistributed(Dense(out_size, activation='softmax')))

    model.summary()


    # Train the model 

    model.compile('rmsprop', 'categorical_crossentropy')

    # define callback function, checkpoint the best models and apply early stopping 
    filepath = "models/NER-ATIS-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
    callbacks_list = [checkpoint, earlystopping]

    model.fit(X_train_enc, y_train_enc, batch_size=batch_size, epochs=epochs,
            validation_data=(X_val_enc, y_val_enc), callbacks=callbacks_list)


    model.save('models/NER_ATIS.h5')


    # Evaluate the model

    model = load_model('models/NER_ATIS.h5')

    # get the test set score (categorical crossentropy / loss), not USEFUL 
    test_score = model.evaluate(X_test_enc, y_test_enc, batch_size=batch_size, verbose=0)
    print('Raw test score:', test_score)

    # shows accuracy of the model on a word level and the confusion matrix of the test set 
    acc, cm = compare_prediction_groundtruth(model, X_test_enc, y_test_enc, idx2la, True, indices=[1,2,3])

    # show evaluation statistics of chosen labels
    labels = [1, 2, 3]
    for label in labels:
        get_evaluation_statistics(idx2la, label)


