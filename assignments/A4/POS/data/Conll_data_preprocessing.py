""" 
Justin Yuan, Aug, 1st, 2017

data description (as in train.txt and test.txt)
- each line is a word with format: word POS_tag chunk_label
- each instance is separated by a '\n\n'
"""

import pickle


def get_corpus(path):
    """ return a list of sample instances of the corpus 
    """
    raw = open(path, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(' ')
        # temp variable to store the line for the current instance
        point.append(stripped_line)

        # reach the end of the current instance
        if line == '\n':
            all_x.append(point[:-1])
            point = []  
    return all_x

def get_maxlen(all_x):
    """ get the maximum length (of text) among all sample instances 
    """
    lengths = [len(x) for x in all_x]
    maxlen = max(lengths)
    return maxlen

def build_data(all_x):
    """ get texts, tags and chunk labels in the samples 
    """
    X = [[c[0] for c in x] for x in all_x]
    tags = [[c[1] for c in y] for y in all_x]
    chunks = [[c[2] for c in z] for z in all_x]
    return X, tags, chunks

def build_word_idx_dicts(X):
    """ build mapping dictionaries between words and indices 
    """
    # get all words in the corpus 
    all_text = [c for x in X for c in x]
    # build the vocabulary 
    words = list(set(all_text))
    word2ind = {word: (index+1) for index, word in enumerate(words)}
    ind2word = {(index+1): word for index, word in enumerate(words)}
    return word2ind, ind2word

def build_label_idx_dicts(tags):
    """ build mapping dictionaries between tags/labels and indices 
    """
    labels = list(set([c for x in tags for c in x]))
    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    ind2label = {(index + 1): label for index, label in enumerate(labels)}
    return label2ind, ind2label



if __name__ == "__main__":
    """ generate training, validation and test set from train.txt and test.txt
    """

    # traing set 
    all_x_train = get_corpus('train.txt')
    maxlen_train = get_maxlen(all_x_train)
    X_train, tags_train, chunks_train = build_data(all_x_train)

    # test set 
    all_x_test = get_corpus('test.txt')
    maxlen_test = get_maxlen(all_x_test)
    X_test, tags_test, chunks_test = build_data(all_x_test)

    # full dataset 
    X = X_train + X_test
    tags = tags_train + tags_test
    chunks = chunks_train + chunks_test

    maxlen = max(maxlen_train, maxlen_test)
    word2ind, ind2word = build_word_idx_dicts(X)
    label2ind, ind2label = build_label_idx_dicts(tags)

    # compile processed data and save
    data = {'full':{}, 'train':{}, 'test':{}, 'stats':{}}

    data['full']['X'] = X
    data['full']['tags'] = tags
    data['full']['chunks'] = chunks

    data['train']['X'] = X_train
    data['train']['tags'] = tags_train
    data['train']['chunks'] = chunks_train

    data['test']['X'] = X_test
    data['test']['tags'] = tags_test
    data['test']['chunks'] = chunks_test

    data['stats']['maxlen'] = maxlen
    data['stats']['word2ind'] = word2ind
    data['stats']['ind2word'] = ind2word
    data['stats']['label2ind'] = label2ind
    data['stats']['ind2label'] = ind2label

    with open('pos_conll.pkl', 'wb') as f:
        pickle.dump(data, f)