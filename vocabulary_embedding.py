import jsonlines
import pickle
from collections import Counter
from itertools import chain
import keras
import matplotlib.pyplot as plt
import numpy as np
import sys

class vocabHandler():
    FN = 'vocabulary_embedding'
    vocab_size=40000
    embedding_dim=100
    lower=False
    glove_n_symbols = 400000
    seed=42
   
   
    # O counter conta as palavras mais comuns, e depois eu ordeno baseado na aparição
    def get_vocab(self, lst) :

        vocabcount = Counter(w for txt in lst for w in txt.split())
        vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
        return vocab

    def get_idx(self, vocab):
        empty = 0
        eos = 1
        start_idx = eos + 1
        word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab)) # word2idx['the'] = 0
        word2idx['<empty>'] = empty
        word2idx['<eos>'] = eos

        idx2word = dict( (idx, word) for word,idx in word2idx.items()) # idx2word[0] = 'the'
        return word2idx, idx2word

    def get_index_weights(self):
        glove_index_dict = {}
        glove_embedding_weights = np.empty((self.glove_n_symbols, self.embedding_dim))
        glove_name = 'dataset/glove.6B.%dd.txt'%self.embedding_dim
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

        for w,i in glove_index_dict.items():
            w = w.lower()
            if w not in glove_index_dict:
                glove_index_dict[w] = i

        return glove_index_dict, glove_embedding_weights


    def get_emb_matrix(self, idx2word, glove_index_dict, glove_embedding_weights):
        np.random.seed(self.seed)
        shape = (self.vocab_size, self.embedding_dim)
        scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
        embedding = np.random.uniform(low=-scale, high=scale, size=shape)
        print ('random-embedding/glove scale', scale, 'std', embedding.std())

        # copy from glove weights of words that appear in our short vocabulary (idx2word)
        c = 0

        for i in range(self.vocab_size):
            w = idx2word[i]
            g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
            if g is None and w.startswith('#'): 
                w = w[1:]
                g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
            if g is not None:
                embedding[i,:] = glove_embedding_weights[g,:]
                c+=1
            #print(embedding[i,:])
        print ('number of tokens, in small vocab, found in glove and copied to embedding', c,c/float(self.vocab_size))

        return embedding

    def get_gloveidx2idx(self, word2idx, idx2word, glove_index_dict, glove_embedding_weights, embedding, vocab_size):
        glove_thr = 0.5
        word2glove = {}
        for w in word2idx:
            if w in glove_index_dict:
                g = w
            elif w.lower() in glove_index_dict:
                g = w.lower()
            elif w.startswith('#') and w[1:] in glove_index_dict:
                g = w[1:]
            elif w.startswith('#') and w[1:].lower() in glove_index_dict:
                g = w[1:].lower()
            else:
                continue
            word2glove[w] = g

        normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]

        nb_unknown_words = 100

        glove_match = []
        for w,idx in word2idx.items():
            if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
                gidx = glove_index_dict[word2glove[w]]
                gweight = glove_embedding_weights[gidx,:].copy()
                # find row in embedding that has the highest cos score with gweight
                gweight /= np.sqrt(np.dot(gweight,gweight))
                score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
                while True:
                    embedding_idx = score.argmax()
                    s = score[embedding_idx]
                    if s < glove_thr:
                        break
                    if idx2word[embedding_idx] in word2glove :
                        glove_match.append((w, embedding_idx, s)) 
                        break
                    score[embedding_idx] = -1
        glove_match.sort(key = lambda x: -x[2])
        print ('# of glove substitutes found', len(glove_match))

        #for orig, sub, score in glove_match[-10:]:
        #    print (score, orig,'=>', idx2word[sub])

        glove_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in glove_match)

        return glove_idx2idx

    def get_X(self, word2idx, desc):
        X = [[word2idx[token] for token in d.split()] for d in desc]
        #plt.hist(map(len,X),bins=50);
        return X

    def get_Y(self, word2idx, heads):
        Y = [[word2idx[token] for token in headline.split()] for headline in heads]
        #plt.hist(map(len,Y),bins=50);
        return Y

    def parse_dataset(self):
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
                if j == 600000 :
                    break

        print("")
        print("")

        vocab = self.get_vocab(heads + desc)

        word2idx, idx2word = self.get_idx(vocab)

        glove_index_dict, glove_embedding_weights = self.get_index_weights()

        embedding = self.get_emb_matrix(idx2word, glove_index_dict, glove_embedding_weights)

        vocab_size, embedding_size = embedding.shape

        glove_idx2idx = self.get_gloveidx2idx(word2idx, idx2word, glove_index_dict, glove_embedding_weights, embedding, vocab_size)

        #print("")
        #print(glove_idx2idx)
        #print("")

        X = self.get_X(word2idx, desc)
        Y = self.get_Y(word2idx, heads)

        return embedding, idx2word, word2idx, X, Y, glove_idx2idx
        #return embMatrix

    def prt(self, label, X, idx2word):
        print (label+':')
        for w in X:
            sys.stdout.write(idx2word[w] + " ")
        print()


