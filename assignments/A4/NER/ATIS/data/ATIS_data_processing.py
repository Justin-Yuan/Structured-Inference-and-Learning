"""
Justin Yuan, Aug 1st, 2017

Data preprocessing code for ATIS datasets

all dicts in different files are the same
"""
# coding: utf-8

import pickle 


# combine all data sets 
train_set_full = [[],[],[]]
valid_set_full = [[],[],[]]
test_set_full = [[],[],[]]
dicts_full = {}
get_dict = False

for i in range(5):
    filename = 'ATIS data sets/atis.fold'+str(i)+'.pkl'
    with open(filename, 'rb') as f:
        train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')
    for i in range(len(train_set)):
        train_set_full[i] += train_set[i]
        valid_set_full[i] += valid_set[i]
        test_set_full[i] += test_set[i]
    if not get_dict:
        dicts_full = dicts
        get_dict = True
        
data_full = (tuple(train_set_full), tuple(valid_set_full), tuple(test_set_full), dicts_full)


# Save and reload full data set

with open('atis.pkl', 'wb') as f:
    pickle.dump(data_full, f)


