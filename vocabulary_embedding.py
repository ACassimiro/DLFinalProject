import jsonlines
import pickle
from collections import Counter
from itertools import chain
import keras
import matplotlib.pyplot as plt
import numpy as np

FN = 'vocabulary_embedding'
seed=42
vocab_size=40000
embedding_dim=100
lower=False
glove_n_symbols = 400000

# O counter conta as palavras mais comuns, e depois eu ordeno baseado na aparição
def get_vocab(lst) :
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab

def get_idx(vocab) :
    empty = 0
    eos = 1
    start_idx = eos + 1
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab)) # word2idx['the'] = 0
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos

    idx2word = dict( (idx, word) for word,idx in word2idx.items()) # idx2word[0] = 'the'
    return word2idx, idx2word

FN0 = "sample-1M"
heads = []
desc = []
j = 0

with jsonlines.open('dataset/%s.jsonl'%FN0, 'r') as reader:
    for obj in reader :
        heads.append( obj['title'] )
        desc.append( obj['content'])
        j = j + 1
        if j % 50000 == 0:
            print(j)
        if j == 10 :
            break

vocab = get_vocab(heads + desc)

word2idx, idx2word = get_idx(vocab)

glove_index_dict = {}
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
glove_name = 'dataset/glove.6B.%dd.txt'%embedding_dim
global_scale = .1
with open(glove_name, 'r') as fp :
    i = 0
    for l in fp :
        l = l.strip().split()
        w = l[0]
        glove_index_dict[w] = i # glove_index_dict['dog'] = 9
        glove_embedding_weights[i, :] = [float(x) for x in l[1:]] # glove_embedding_weights
        i += 1

glove_embedding_weights *= global_scale
print(glove_embedding_weights.std())
print(glove_index_dict['of'])
