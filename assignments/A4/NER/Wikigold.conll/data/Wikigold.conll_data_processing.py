"""
Justin Yuanm, Aug 1st, 2017

the data preprocessing procedures here are very similar to that in POS data preprocessing,
the script is written in a sequential flow for convenience.
"""
# coding: utf-8

import pickle 


# Process the data 
raw = open('wikigold.conll.txt', 'r').readlines()
all_x = []
point = []

for line in raw:
    stripped_line = line.strip().split(' ')
    point.append(stripped_line)
    if line == '\n':
        all_x.append(point[:-1])
        point = []     

all_x = all_x[:-1]
lengths = [len(x) for x in all_x]
print(max(lengths))
short_x = [x for x in all_x if len(x) < 64]

X = [[c[0] for c in x] for x in short_x]
y = [[c[1] for c in y] for y in short_x]

all_text = [c for x in X for c in x]
words = list(set(all_text))
word2ind = {word: (index+1) for index, word in enumerate(words)}
ind2word = {(index+1): word for index, word in enumerate(words)}

labels = list(set([c for x in y for c in x]))
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}

maxlen = max([len(x) for x in X])
print('Maximum sequence length:', maxlen)   # 63 is correct

# without data shuffling
train_test_split = 0.1
test_num = int(len(X) * train_test_split)
X_train = X[:-test_num]
y_train = y[:-test_num]
X_test = X[-test_num:]
y_test = y[-test_num:]

#  Save the processed data
data = {'train':{}, 'test':{}, 'stats':{}}
data['train']['X'] = X_train
data['train']['y'] = y_train
data['test']['X'] = X_test 
data['test']['y'] = y_test 
data['stats']['maxlen'] = maxlen 
data['stats']['word2ind'] = word2ind
data['stats']['ind2word'] = ind2word
data['stats']['label2ind'] = label2ind
data['stats']['ind2label'] = ind2label

with open('conll.pkl', 'wb') as f:
    pickle.dump(data, f)

