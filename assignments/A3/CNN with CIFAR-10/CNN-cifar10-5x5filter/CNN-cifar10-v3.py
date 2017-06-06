
# coding: utf-8

# # Output layer weights 

# In[1]:

import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar10
from keras.models import Sequential 
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# get_ipython().magic('matplotlib inline')


# ## Data acquisition & hyperparameter setting

# In[2]:

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True


# In[3]:

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[4]:

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ## Build model 

# In[7]:

model = Sequential()

model.add(Conv2D(64, (5, 5), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()


# ## Compile Model 

# In[50]:

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


# In[51]:

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train.shape


# In[52]:

print('Not using data augmentation.')
model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=epochs,
         validation_data=(x_test, y_test),
         shuffle=True)


# ## Retrieve weights from Layers 

# In[63]:

# print(len(model.layers[0].get_weights()))
# print(model.layers[0].get_weights()[0].shape)
# print(model.layers[0].get_weights()[1].shape)
# conv1_weights = model.layers[0].get_weights()[0]
# conv1_biases = model.layers[0].get_weights()[1]
# model.layers[0].get_weights()[0]


# # ## Visualize layers 

# # In[69]:

# def get_filter(weights, biases, index):
#     filter_weights = [weights[i][j][k][index] for i in range(weights.shape[0])                      for j in range(weights.shape[1]) for k in range(weights.shape[2])]
#     return np.asarray(filter_weights).reshape((3,3,3)), biases[index]

# # print(get_filter(conv1_weights, conv1_biases, 0))

# def draw_visualizations(weights, ):
    


# # In[ ]:




# ## Save the model & trained weights 

# In[16]:

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# ## Reload model & weights 

# In[38]:

import h5py 
fpath = "model.h5"
f = h5py.File(fpath, 'r')

