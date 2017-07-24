
# coding: utf-8

# In[14]:

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
from keras.layers import Convolution1D, MaxPooling1D

from keras.callbacks import ModelCheckpoint, EarlyStopping


# ## Load the data

# In[2]:

### Load Data
with open('atis.pkl', 'rb') as f:
    train_set, valid_set, test_set, dicts = pickle.load(f)

w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

# Create index to word/label dicts
idx2w  = {w2idx[k]:k for k in w2idx}
idx2ne = {ne2idx[k]:k for k in ne2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}


# In[3]:

print(len(train_set), len(train_set[0]))
print(len(labels2idx))
print(sorted(list(labels2idx.values())))
print(idx2ne[0])


# In[4]:

train_x, train_ne, train_label = train_set
val_x, val_ne, val_label = valid_set
test_x, test_ne, test_label = test_set

X = train_x + val_x + test_x
ne = train_ne + val_ne + test_ne 
label = train_label + val_label + test_label 

# words_test = [ list(map(lambda x: idx2w[x], w)) for w in test_x]
# groundtruth_test = [ list(map(lambda x: idx2la[x], y)) for y in test_label]
# words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
# groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]
# words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
# groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]


# In[5]:

print(type(X), type(label))
print(X[0])
print(label[0])
for i in range(5):
    print(len(X[i]))
    print(X[i])
print()
for i in range(5):
    print(len(label[i]))
    print(label[i])


# In[6]:

maxlen = max([len(x) for x in X])
print('Maximum sequence length:', maxlen)


# In[7]:

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result


# In[62]:

max_features = max(list(w2idx.values()))
print(max_features)

X_enc = pad_sequences(X, maxlen=maxlen, value=max_features+1)


# In[63]:

print(type(X_enc))
print(X_enc.shape)
print(X_enc[:2])


# In[64]:

max_label = max(labels2idx.values()) + 1
print(max_label)

y_enc = [[0] * (maxlen - len(ey)) + [ey] for ey in label]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
y_enc = pad_sequences(y_enc, maxlen=maxlen)


# In[65]:

print(y_enc[0])
print(len(y_enc[0]))
print(type(y_enc))
print(y_enc.shape)


# In[66]:

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


# In[67]:

print('Training and testing tensor shapes:')
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)


# ## Build the model

# In[68]:

# n_classes = len(idx2la)
# n_vocab = len(idx2w)

max_features = len(w2idx)+1
embedding_size = 100
hidden_size = 32
out_size = len(labels2idx)
batch_size = 32
epochs = 10


# In[69]:

# Define model
model = Sequential()
# model.add(Embedding(n_vocab,100))
model.add(Embedding(input_dim=max_features, output_dim=embedding_size,
                    input_length=maxlen, mask_zero=False))
model.add(Convolution1D(64,5,padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(GRU(100,return_sequences=True))
model.add(TimeDistributed(Dense(out_size, activation='softmax')))

model.summary()


# ## Train the model 

# In[70]:

model.compile('rmsprop', 'categorical_crossentropy')


# In[71]:

filepath = "models/NER-ATIS-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
callbacks_list = [checkpoint, earlystopping]


# In[72]:

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(X_val, y_val), callbacks=callbacks_list)


# ## Evaluate the model

# In[41]:

model = load_model('models/NER-ATIS-09-0.19.hdf5')
model.summary()


# In[42]:

print(X_test[:2])
score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print('Raw test score:', score)


# In[20]:

def score(yh, pr):
    coords = [np.where(yhh < max_features+1)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr


# In[27]:

# On the validation set 

pr = model.predict(X_val)
pr = pr.argmax(2)
print(pr.shape)
print(pr[0])
print(pr[0][0])
yh = y_test.argmax(2)
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


# In[26]:

fyh, fpr = score(yh, pr)
print('Testing accuracy:', accuracy_score(fyh, fpr))
print('Testing confusion matrix:')
print(confusion_matrix(fyh, fpr))

