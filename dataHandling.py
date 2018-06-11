from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
import numpy as np
import random
import sys


class dataHandler():
    oov0 = 0
    maxlend = 0
    maxlenh = 0
    maxlen = 0
    glove_idx2idx = []
    empty  = 0
    eos  =  1
    batch_size = 100
    seed = 42
    vocab_size = 0
    nb_unknown_words = 0

    def __init__(self, oov0, glove_idx2idx, maxlend, maxlenh, vocab_size, nb_unknown_words):
        self.oov0 = oov0
        self.glove_idx2idx = glove_idx2idx
        self.maxlend = maxlend
        self.maxlenh = maxlenh
        self.vocab_size = vocab_size
        self.nb_unknown_words = nb_unknown_words
        self.maxlen = maxlend + maxlenh

    def lpadd(self, x, maxlend=maxlend, eos=eos):
        """left (pre) pad a description to maxlend and then add eos.
        The eos is the input to predicting the first word in the headline
        """
        assert maxlend >= 0
        if maxlend == 0:
            return [eos]
        n = len(x)
        if n > maxlend:
            x = x[-maxlend:]
            n = maxlend
        return [self.empty]*(maxlend-n) + x + [eos]


    def vocab_fold(self, xs):
        """convert list of word indexes that may contain words outside vocab_size to words inside.
        If a word is outside, try first to use glove_idx2idx to find a similar word inside.
        If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
        """
        #print("")
        #print(self.glove_idx2idx)
        #print("")

        xs = [x if x < self.oov0 else self.glove_idx2idx.get(x,x) for x in xs]
        # the more popular word is <0> and so on
        outside = sorted([x for x in xs if x >= self.oov0])
        # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
        outside = dict((x,self.vocab_size-1-min(i, self.nb_unknown_words-1)) for i, x in enumerate(outside))
        xs = [outside.get(x,x) for x in xs]
        return xs


    def flip_headline(self, x, nflips=None, model=None, debug=False):
        """given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline)
        with words predicted by the model
        """
        if nflips is None or model is None or nflips <= 0:
            return x
        
        batch_size = len(x)
        assert np.all(x[:,self.maxlend] == self.eos)

        print("")
        print("ANTES")

        probs = model.predict(x, verbose=0, batch_size=batch_size)
        
        print("DEPOIS")
        print("")
        
        x_out = x.copy()
        for b in range(batch_size):
            # pick locations we want to flip
            # 0...maxlend-1 are descriptions and should be fixed
            # maxlend is eos and should be fixed
            flips = sorted(random.sample(range(self.maxlend+1,self.maxlen), nflips))
            if debug and b < debug:
                print (b)
            for input_idx in flips:
                if x[b,input_idx] == self.empty or x[b,input_idx] == eos:
                    continue
                # convert from input location to label location
                # the output at maxlend (when input is eos) is feed as input at maxlend+1
                label_idx = input_idx - (self.maxlend+1)
                prob = probs[b, label_idx]
                w = prob.argmax()
                if w == self.empty:  # replace accidental empty with oov
                    w = self.oov0
                if debug and b < debug:
                    print ('%s => %s'%(idx2word[x_out[b,input_idx]],idx2word[w]))
               
                x_out[b,input_idx] = w
                
            if debug and b < debug:
                print()
        return x_out


    def  conv_seq_labels(self, xds, xhs, nflips=None, model=None, debug=False):
        """description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
        batch_size = len(xhs)
        assert len(xds) == batch_size
        x = [self.vocab_fold(self.lpadd(xd, self.maxlend, self.eos)+xh) for xd,xh in zip(xds,xhs)]  # the input does not have 2nd eos
        x = sequence.pad_sequences(x, maxlen=self.maxlen, value=self.empty, padding='post', truncating='post')
        x = self.flip_headline(x, nflips=nflips, model=model, debug=debug)
        
        y = np.zeros((batch_size, self.maxlenh, self.vocab_size))
        for i, xh in enumerate(xhs):
            xh = self.vocab_fold(xh) + [self.eos] + [self.empty]*self.maxlenh  # output does have a eos at end
            xh = xh[:self.maxlenh]
            y[i,:,:] = to_categorical(xh, self.vocab_size)
            
        return x, y


    def gen(self, Xd,  Xh,  batch_size = batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
        """yield batches. for training use nb_batches=None
        for validation generate deterministic results repeating every nb_batches
        
        while training it is good idea to flip once in a while the values of the headlines from the
        value taken from Xh to value generated by the model.
        """
        c = nb_batches if nb_batches else 0
        while True:
            xds = []
            xhs = []
            if nb_batches and c >= nb_batches:
                c = 0
            new_seed = random.randint(0, sys.maxsize)
            random.seed(c+123456789+seed)
            for b in range(batch_size):
                t = random.randint(0,len(Xd)-1)

                xd = Xd[t]
                s = random.randint(min(self.maxlend,len(xd)), max(self.maxlend,len(xd)))
                xds.append(xd[:s])
                
                xh = Xh[t]
                s = random.randint(min(self.maxlenh,len(xh)), max(self.maxlenh,len(xh)))
                xhs.append(xh[:s])

            # undo the seeding before we yield inorder not to affect the caller
            c+= 1
            random.seed(new_seed)

            yield self.conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)