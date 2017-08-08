
# coding: utf-8

# ## Practical 2: Text Classification with Word Embedding
# <p>Oxford CS - Deep NLP 2017<br>
# https://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/</p>
# <p>[Yannis Assael, Brendan Shillingford, Chris Dyer]</p>

# In[1]:

import numpy as np
import time
import os
from random import shuffle
import re


# In[2]:

from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
output_notebook()


# ### Load TED dataset 

# In[3]:

import urllib.request
import zipfile
import lxml.etree


# In[4]:

# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")


# In[5]:

with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

# get all text body 
doc_list = doc.xpath('//content/text()')

# get all keywords / labels 
label_list = doc.xpath('//keywords/text()')

del doc


# In[6]:

def get_label(keywords, label_dict):
    """ map each document into a proper label from its keywords 
    """
    label_string = keywords.lower()
    if ("technology" in label_string) and ("entertainment" in label_string) and ("design" in label_string):
        return label_dict['TED']
    elif ("entertainment" in label_string) and ("design" in label_string):
        return label_dict['oED']
    elif ("technology" in label_string) and ("design" in label_string):
        return label_dict['ToD']
    elif ("technology" in label_string) and ("entertainment" in label_string):
        return label_dict['TEo']
    elif ("design" in label_string):
        return label_dict['ooD']
    elif ("entertainment" in label_string):
        return label_dict['oEo']
    elif ("technology" in label_string):
        return label_dict['Too']
    else:
        return label_dict['ooo']


# In[7]:

# proper labels as specified 
labels = ['ooo', 'Too', 'oEo', 'ooD', 'TEo', 'ToD', 'oED', 'TED']
label_dict = {labels[i]: i for i in range(8)}

# get all proper labels for the documents 
label_list_temp = [get_label(keywords, label_dict) for keywords in label_list]   

# a list of (text body, label) document samples 
labelled_doc = list(zip(doc_list, label_list_temp))


# ### Split into training, validation and test sets

# In[8]:

from random import sample

def divide_dataset(labelled_doc, num_train, num_valid, num_test=None, shuffle=False):
    """ construct training, validation and test set given number of samples 
    """
    if num_test == None:
        num_test = len(labelled_doc) - num_train - num_valid
        
    if shuffle:
        temp = sample(labelled_doc, len(labelled_doc))
    else:
        temp = labelled_doc
    return temp[:num_train], temp[num_train:-num_test], temp[-num_test:]


# In[9]:

train_doc_temp, valid_doc_temp, test_doc_temp = divide_dataset(labelled_doc, 1585, 250, 250, shuffle=True)

print("training set size:",len(train_doc_temp))
print("validation set size:", len(valid_doc_temp))
print("test set size", len(test_doc_temp))


# ## Build vocabulary using training set 

# In[10]:

from collections import Counter 

def tokenize_and_lowercase(text):
    """ taken from assignment 1,
        return a list of sentences, each sentence is a list of words 
    """
    text_noparens = re.sub(r'\([^)]*\)', '', text)
    sentences_strings = []
    for line in text_noparens.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_strings.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
        
    sentences= []
    for sent_str in sentences_strings:
        tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
        sentences.append(tokens)
    return sentences

def get_most_common_words_list(sentences, num_words):
    """ return a list of most common words 
    """
    counts_ted_top1000 = []
    c = Counter([word for sent in sentences for word in sent])
    list_most_common = c.most_common(num_words)
    words_most_common = [item[0] for item in list_most_common]
    #for word, count in list_most_common:
    #    counts_ted_top1000.append(count)
    return words_most_common
    
def replace_unknown_token(sent_list, words_most_common, unknown_token="UNK"):
    """ filter a list of words, replace unknown words with the unknown token 
    """
    filtered_list = [word if word in words_most_common else unknown_token for word in sent_list]  # so fast !!!
    return filtered_list

def tokenize_and_lowercase_most_common(text, words_most_common):
    """ return a list of sentences as lists of words that are most common (in the most common word list)
    """
    text_noparens = re.sub(r'\([^)]*\)', '', text)
    sentences_strings = []
    for line in text_noparens.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_strings.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
        
    sentences= []
    for sent_str in sentences_strings:
        tokens = replace_unknown_token(re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split(), words_most_common)
        if tokens != []:
            sentences.append(tokens)
    return sentences

def build_dataset(doc_temp, words_most_common):
    """ apply filtering (tokenization, lower-case convertion and common words selection) to each text body in the corpus
    """
    doc_new = [(tokenize_and_lowercase_most_common(doc[0], words_most_common), doc[1]) for doc in doc_temp]
    doc_final = [item for item in doc_new if item[0] != []]
    return doc_final


# In[11]:

input_text = '\n'.join([train_doc_temp[i][0] for i in range(len(train_doc_temp))])
sentences_ted = tokenize_and_lowercase(input_text)
# train_doc = [(tokenize_and_lowercase(doc_temp[0]), doc_temp[1]) for doc_temp in train_doc_temp]

# Get the list of 9999 most common words 
words_most_common = get_most_common_words_list(sentences_ted, 9999)

# build a temporary training set for Word2Vec embedding
train_doc = [(tokenize_and_lowercase_most_common(doc_temp[0], words_most_common), doc_temp[1]) for doc_temp in train_doc_temp]


# ### Rebuild the vocabulary (add unknown token to the vocabulary)

# In[12]:

def rebuild_vocab(train_doc):
    """ filtered out unknown tokens, rebuild the dataset with the top vocabulary,
        return a list of sentences that can be used to train Word2Vec embedding.
    """
    sentences = [sent for doc in train_doc for sent in doc[0]]
    return sentences 


# In[13]:

sentences = rebuild_vocab(train_doc)


# ## Model 

# ### Word2Vec Embedding

# In[20]:

import os 
from gensim.models import Word2Vec

def build_word2vec_model(name, sentences=None, min_count=10, size=100):
    """ train a Word2Vec embedding model or load an existing one 
    """
    model = Word2Vec(sentences, min_count=min_count, size=size)
    model.save(name)
#     if not os.path.isfile(name):
#         model = Word2Vec(sentences, min_count=min_count, size=size)
#         model.save(name)
#     else:
#         model = Word2Vec.load(name)
    return model 


# In[21]:

model = build_word2vec_model('word2vec_model', sentences=sentences, min_count=10, size=100)


# ## Datasets

# In[22]:

# Training set 
train_doc = build_dataset(train_doc_temp, words_most_common)

# Validation set 
valid_doc = build_dataset(valid_doc_temp, words_most_common)

# Test set 
test_doc = build_dataset(test_doc_temp, words_most_common)


# In[23]:

np.savez('corpus_all_9999', train_doc=train_doc, valid_doc=valid_doc, test_doc=test_doc)


# In[24]:

def embed_text(model, text):
    """ embed the input text as a model vector 
    
    Arguments:
        model: Word2Vec model.
        text: input text
    
    Outputs:
        embedded vector 
    """
    vector_list = [model.wv[word] for sent in text for word in sent]
    return sum(vector_list) / len(vector_list)
    
def embed_corpus(model, corpus):
    """ apply embed_text to each text body in the corpus 
    """
    return np.asarray([embed_text(model, doc[0]) for doc in corpus])

def encode_label(label, size):
    """ apply one-hot encoding to the label 
    """
    l = [0]*size
    l[label] = 1
    return l

def encode_class(corpus, size):
    """ apply encode_label to each sample label in the corpus 
    """
    return np.asarray([encode_label(doc[1], size) for doc in corpus])

def embedded_with_class(model, doc, size):
    doc_x = embed_corpus(model, doc)
    doc_y = encode_class(doc, size)
    return doc_x, doc_y


# In[25]:

# get processed datasets (text embedded and label encoded )

train_doc_embed_with_class = embedded_with_class(model, train_doc, len(label_dict))

valid_doc_embed_with_class = embedded_with_class(model, valid_doc, len(label_dict))

test_doc_embed_with_class = embedded_with_class(model, test_doc, len(label_dict))


# In[34]:

np.savez('embedded_corpus_with_labels_all_9999', 
         train_doc_embed_text=train_doc_embed_with_class[0], train_doc_embed_label=train_doc_embed_with_class[1],
         valid_doc_embed_text=valid_doc_embed_with_class[0], valid_doc_embed_label=valid_doc_embed_with_class[1],
         test_doc_embed_text=test_doc_embed_with_class[0], test_doc_embed_label=test_doc_embed_with_class[1])


# ## Bag of Means  model 

# In[35]:

import tensorflow as tf 


# ### model hyperparameters

# In[79]:

epoch = 3000
learning_rate = 0.001
batch_size = 50
total_batch = int(train_doc_embed_with_class[0].shape[0] / batch_size)
index = 0


# ### data placeholders 

# In[64]:

x = tf.placeholder(tf.float32, shape=[None, 100])
y = tf.placeholder(tf.int32, shape=[None, 8])


# ### model trainable parameters 

# In[65]:

W = tf.Variable(tf.truncated_normal(shape=[100, 256]))
b = tf.Variable(tf.constant(0.0, shape=[256]))

V = tf.Variable(tf.truncated_normal(shape=[256, 8]))
c = tf.Variable(tf.constant(0.0, shape=[8]))


# ### model architecture 

# In[66]:

h = tf.tanh(tf.matmul(x, W) + b)
u = tf.matmul(h, V) + c

p = tf.nn.softmax(u)
pred = tf.argmax(p, 1)


# In[67]:

loss = tf.reduce_mean(tf.reduce_sum(-tf.cast(y, tf.float32)*tf.log(tf.clip_by_value(p, 1e-10, 1.0)), 1))
# need to clip the values for stable training experimentally  
# loss = tf.reduce_mean(tf.reduce_sum(-tf.cast(y, tf.float32)*tf.log(p), 1))

accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(y, 1)), tf.float32))


# In[68]:

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[69]:

# instantiate model saver to save the model in a session 
saver = tf.train.Saver()


# In[70]:

def next_batch(data, index, size):
    """ return next batch in format: index, x batch, y batch
    """
    if index + size <= data[0].shape[0]:
        return index+size, data[0][index:index+size], data[1][index:index+size]
    else:
        return index+size-data[0].shape[0], np.concatenate((data[0][index:],data[0][:index+size-data[0].shape[0]]), 0),     np.concatenate((data[1][index:],data[1][:index+size-data[1].shape[0]]), 0)


# ### Training 

# In[80]:

sess = tf.InteractiveSession()


# In[81]:

# initialize all trainable model variables 
init = tf.global_variables_initializer()
sess.run(init)


# In[82]:

# start training 
for i in range(epoch):
    xloss = 0
    
    for j in range(total_batch):
        index, x_, y_ = next_batch(train_doc_embed_with_class, index, batch_size)
        _, xloss, acc_train = sess.run([optimizer, loss, accuracy], feed_dict={x: x_, y: y_})
        
#         if j % 10 == 0:
#             print("epoch %d, run %d, loss %g" % (i, j, xloss))
            
    if i % 100 == 0:
        acc_val = sess.run(accuracy, feed_dict={x:valid_doc_embed_with_class[0], y:valid_doc_embed_with_class[1]})
        print("epoch %d, Training acc: %g, Validation acc: %g " % (i, acc_train, acc_val))
        
save_path = saver.save(sess, "models/model.ckpt")
print("Model saved in file: %s" % save_path)

sess.close()


# ### Test

# In[83]:

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "models/model.ckpt")
    print("Model restored.")

    acc = sess.run(accuracy, feed_dict={x:test_doc_embed_with_class[0], y:test_doc_embed_with_class[1]})
    print("Test acc: %g" % (acc))


# ## Questions to answer 

# - Compare the learning curves of the model starting from random embeddings, starting from GloVe embeddings (http://nlp.stanford.edu/data/glove.6B.zip; 50 dimensions) or fixed to be the GloVe values. Training in batches is more stable (e.g. 50), which model works best on training vs. test? Which model works best on held-out accuracy?
# - What happens if you try alternative non-linearities (logistic sigmoid or ReLU instead of tanh)?
# - What happens if you add dropout to the network?
# - What happens if you vary the size of the hidden layer?
# - How would the code change if you wanted to add a second hidden layer?
# - How does the training algorithm affect the quality of the model?
# - Project the embeddings of the labels onto 2 dimensions and visualise (each row of the projection matrix V corresponds a label embedding). Do you see anything interesting?

# ## Playground 

# In[55]:

# trainable parameters 
W = tf.Variable(tf.truncated_normal(shape=[100, 256]))
b = tf.Variable(tf.constant(0.0, shape=[256]))

W2 = tf.Variable(tf.truncated_normal(shape=[256, 128]))
b2 = tf.Variable(tf.constant(0.0, shape=[128]))

V = tf.Variable(tf.truncated_normal(shape=[128, 8]))
c = tf.Variable(tf.constant(0.0, shape=[8]))

# additional placeholders 
dropout_rate = tf.placeholder(tf.float32)

# model architecture 
h = tf.nn.relu(tf.matmul(x, W) + b)
h2 = tf.nn.relu(tf.matmul(h, W2) + b2)
h2_drop = tf.nn.dropout(h2, keep_prob=dropout_rate)
u = tf.matmul(h2_drop, V) + c
p = tf.nn.softmax(u)
pred = tf.argmax(p, 1)

# training preparations 
loss = tf.reduce_mean(tf.reduce_sum(-tf.cast(y, tf.float32)*tf.log(tf.clip_by_value(p, 1e-10, 1.0)), 1))

accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(y, 1)), tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()


# In[56]:

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)


# In[57]:

for i in range(epoch):
    xloss = 0
    acc = 0.0
    
    for j in range(total_batch):
        index, x_, y_ = next_batch(train_doc_embed_with_class, index, batch_size)
        _, xloss = sess.run([optimizer, loss], feed_dict={x: x_, y: y_, dropout_rate: 0.5})
        
        if j % 30 == 0:
            print("epoch %d, run %d, loss %g" % (i, j, xloss))
            
    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x:test_doc_embed_with_class[0], y:test_doc_embed_with_class[1], dropout_rate: 1.0})
        print("epoch %d, Test acc: %g" % (i, acc * 100), end="")
        print("%")
        
save_path = saver.save(sess, "models/model"+str(time.time())+".ckpt")
print("Model saved in file: %s" % save_path)  

sess.close()

