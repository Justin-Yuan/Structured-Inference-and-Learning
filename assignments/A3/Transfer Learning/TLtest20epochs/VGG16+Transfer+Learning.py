
# coding: utf-8

# # CNN Transfer Learning with VGG16

# In[60]:

import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

import keras
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

from keras.layers import Input, Flatten, Dense
from keras.models import Model


# In[61]:

from keras import applications

input_img = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
print(input_img)

# model = VGG16(weights='imagenet', input_tensor=input_img)
model_vgg16 = VGG16(weights='imagenet', include_top=False)


# In[62]:

model_vgg16.summary()


# In[63]:

from keras.datasets import cifar10

num_classes = 10
batch_size = 32
epochs = 20

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[64]:

image_size = 32
channel = 3
input_layer = Input(shape=(image_size, image_size, channel), name='image_input')


# In[65]:

# take the output from chopped-off vgg16 model
output_model_vgg16 = model_vgg16(input_layer)

x = Flatten(name='flatten')(output_model_vgg16)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dense(10, activation='softmax', name='predictions')(x)

new_model = Model(input=input_layer, output=x)
new_model.summary()


# In[66]:

layer_count = 0
for layer in new_model.layers:
    layer_count = layer_count + 1
    print(layer)


# In[67]:

# fix pretrained weights in vgg16 for training
for i in range(layer_count-3):
    new_model.layers[i].trainable = False


# ## Train the new model

# In[68]:

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

new_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[69]:

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[ ]:

new_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
             validation_data=(x_test, y_test), shuffle=True)


# In[ ]:

new_model.save('new_model.h5')

new_model.save_weights('new_model_weights.h5')

print("new model saved")
