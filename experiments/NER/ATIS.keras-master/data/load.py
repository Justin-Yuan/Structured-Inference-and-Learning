from __future__ import print_function
import gzip
try: #python2
    import cPickle as pickle
    from urllib import urlretrieve
except ImportError: #python3
    import pickle
    from urllib.request import urlretrieve

import os
import random
from os.path import isfile

PREFIX = os.getenv('ATISDATA', '')

def download(origin):
    '''
    download the corresponding atis file
    from http://www-etud.iro.umontreal.ca/~mesnilgr/atis/
    '''
    print('Downloading data from %s' % origin)
    name = origin.split('/')[-1]
    urlretrieve(origin, name)

def download_dropbox():
    ''' 
    download from drop box in the meantime
    '''
    print('Downloading data from https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0')
    os.system('wget -O atis.pkl https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0')

def load_dropbox(filename):
    if not isfile(filename):
        #download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/'+filename)
        download_dropbox()
    #f = gzip.open(filename,'rb')
    f = open(filename,'rb')
    return f

def load_udem(filename):
    if not isfile(filename):
        download('http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/'+filename)
    f = gzip.open(filename,'rb')
    return f


def atisfull():
    f = load_dropbox(PREFIX + 'atis.pkl')
    
    try:
        train_set, test_set, dicts = pickle.load(f)
    except UnicodeDecodeError:
        train_set, test_set, dicts = pickle.load(f, encoding='latin1')
    return train_set, test_set, dicts

def atisfold(fold):
    assert fold in range(5)
    f = load_udem(PREFIX + 'atis.fold'+str(fold)+'.pkl.gz')
    try:
        train_set, valid_set, test_set, dicts = pickle.load(f)
    except UnicodeDecodeError:
        train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')

    return train_set, valid_set, test_set, dicts
 
if __name__ == '__main__':
    
    ''' visualize a few sentences '''

    import pdb
    
    w2ne, w2la = {}, {}
    train, test, dic = atisfull()
    train, _, test, dic = atisfold(1)
    
    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']
    
    idx2w  = {w2idx[k]:k for k in w2idx}
    idx2ne = {ne2idx[k]:k for k in ne2idx}
    idx2la = {labels2idx[k]:k for k in labels2idx}

    test_x,  test_ne,  test_label  = test
    train_x, train_ne, train_label = train
    wlength = 35

    for e in ['train','test']:
      for sw, se, sl in zip(eval(e+'_x'), eval(e+'_ne'), eval(e+'_label')):
        print('WORD'.rjust(wlength), 'LABEL'.rjust(wlength))
        for wx, la in zip(sw, sl): print(idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength))
        print('\n'+'**'*30+'\n')
        pdb.set_trace()
