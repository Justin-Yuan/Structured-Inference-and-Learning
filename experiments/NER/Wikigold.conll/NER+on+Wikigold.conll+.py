
# coding: utf-8

# Reference: https://gist.github.com/dirko/1d596ca757a541da96ac3caa6f291229

# In[5]:

import pickle 
import numpy as np 

# from sklearn.cross_validation import train_test_split
# from lambdawithmask import Lambda as MaskLambda
from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

# from keras.layers.recurrent import LSTM
# from keras.layers.core import Activation, Dense, Input
# from keras.layers.embeddings import Embedding
# from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras.layers import Input, Dense, TimeDistributed
from keras.layers import Embedding, Activation
from keras.layers import GRU, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping


from keras.backend import tf


# ## Load the data

# In[6]:

with open('conll.pkl', 'rb') as f:
    data = pickle.load(f)


# In[7]:

X = data['X']
y = data['y']
word2ind = data['word2ind']
ind2word = data['ind2word']
label2ind = data['label2ind']
ind2label = data['ind2label']


# In[8]:

print(len(X))
print(len(X[0]))
print(X[0])

print(len(y))
print(len(y[0]))
print(y[0])

print(label2ind)
print(ind2label)


# In[9]:

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result


# In[10]:

maxlen = max([len(x) for x in X])
print('Maximum sequence length:', maxlen)

X_enc = [[word2ind[c] for c in x] for x in X]
# X_enc_reverse = [[c for c in reversed(x)] for x in X_enc]
X_enc = pad_sequences(X_enc, maxlen=maxlen)
# X_enc_b = pad_sequences(X_enc_reverse, maxlen=maxlen)


# In[11]:

print(type(X_enc))
print(X_enc.shape)


# In[9]:

max_label = max(label2ind.values()) + 1
print(max_label)

y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
print(y_enc[0])
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
print(len(y_enc[0]))
y_enc = pad_sequences(y_enc, maxlen=maxlen)
print(type(y_enc))
print(y_enc.shape)

# (X_train_f, X_test_f, X_train_b,
#  X_test_b, y_train, y_test) = train_test_split(X_enc_f, X_enc_b, y_enc,
#                                                test_size=11*32, train_size=45*32, random_state=42)


# In[10]:

validation_split = 0.1
test_split = 0.1 

indices = np.arange(X_enc.shape[0])
np.random.shuffle(indices)
X_enc = X_enc[indices]
y_enc = y_enc[indices]
num_validation_samples = int(validation_split * X_enc.shape[0])
num_test_samples = int(test_split * X_enc.shape[0])

X_train = X_enc[:-num_validation_samples-num_test_samples]
y_train = y_enc[:-num_validation_samples-num_test_samples]
X_val = X_enc[-num_validation_samples-num_test_samples:]
y_val = y_enc[-num_validation_samples-num_test_samples:]
X_test = X_enc[-num_test_samples:]
y_test = y_enc[-num_test_samples:]


# In[11]:

print('Training and testing tensor shapes:')
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)


# ## Build the model 

# In[4]:

max_features = len(word2ind)
embedding_size = 128
hidden_size = 32
out_size = len(label2ind) + 1
batch_size = 32
epochs = 30


# In[3]:

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=embedding_size,
                    input_length=maxlen, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))

model.summary()


# ## Train the model 

# In[14]:

model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[19]:

filepath = "models/NER-Wikigold-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
callbacks_list = [checkpoint, earlystopping]


# In[20]:

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(X_val, y_val), callbacks=callbacks_list)


# ## Evaluate the model

# In[21]:

score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Raw test score:', score)


# In[22]:

def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr


# In[29]:

# On the training set ]

pr = model.predict(X_train)
pr = pr.argmax(2)
print(pr.shape)
print(pr[0])
print(pr[0][0])
yh = y_train.argmax(2)
print(yh.shape)
print(yh[0])


# In[30]:

fyh, fpr = score(yh, pr)
print('Testing accuracy:', accuracy_score(fyh, fpr))
print('Testing confusion matrix:')
print(confusion_matrix(fyh, fpr))


# In[27]:

# On the validatiotn set

pr = model.predict(X_val)
pr = pr.argmax(2)
print(pr.shape)
print(pr[0])
print(pr[0][0])
yh = y_val.argmax(2)
print(yh.shape)
print(yh[0])


# In[28]:

fyh, fpr = score(yh, pr)
print('Testing accuracy:', accuracy_score(fyh, fpr))
print('Testing confusion matrix:')
print(confusion_matrix(fyh, fpr))


# In[25]:

# On the test set 
pr = model.predict(X_test)
pr = pr.argmax(2)
print(pr.shape)
print(pr[0])
print(pr[0][0])
yh = y_test.argmax(2)
print(yh.shape)
print(yh[0])


# In[24]:

fyh, fpr = score(yh, pr)
print('Testing accuracy:', accuracy_score(fyh, fpr))
print('Testing confusion matrix:')
print(confusion_matrix(fyh, fpr))

