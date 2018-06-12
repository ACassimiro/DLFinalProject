import keras
import matplotlib.pyplot as plt
from vocabulary_embedding import vocabHandler
from dataHandling import dataHandler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.engine.topology import Layer
from keras.layers import SpatialDropout1D
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from attention_decoder import AttentionDecoder
from keras.layers import RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
from keras import initializers
from keras import regularizers
from keras import constraints
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import numpy as np
import random
import sys

seed = 42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size=64
nflips=10
maxlend=25 # 0 - if we dont want to use description at all
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 32 # must be same as 160330-word-gen
rnn_layers = 1 # match FN1
batch_norm=False
activation_rnn_size = 40 if maxlend else 0
nb_train_samples = 5000
nb_val_samples = 1000
nb_unknown_words = 10

class neuralNetwork():
    def split_sets(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)
        #len(X_train), len(Y_train), len(X_test), len(Y_test)
        return X_train, X_test, Y_train, Y_test

    def create_model(self, embedding, idx2word, word2idx):
        #activation_rnn_size = 40 if maxlend else 0
        vocab_size, embedding_size = embedding.shape
        np.random.seed(seed)
        np.random.seed(seed)

        regularizer = l2(weight_decay) if weight_decay else None

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_size,
                            input_length=maxlen, weights=[embedding],
                            mask_zero=True, name='embedding_1'))
        print("Embedding Layer --- OK!")
        
        #MODELO No 1 
        """
        """
        lstmPreRV = LSTM(rnn_size)
        model.add(lstmPreRV)
        print("LSTM Layer --- OK!")
        
        model.add(RepeatVector(maxlenh))
        print("Repeat Vector --- OK!")
        

        for i in range(rnn_layers):
            lstm = LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                        kernel_regularizer=regularizer, bias_regularizer=regularizer, recurrent_regularizer=regularizer,
                        dropout=0, recurrent_dropout=0,
                        name='lstm_%d'%(i+2))
            model.add(lstm)
            print("LSTM Layer --- OK!")
      
        model.add(TimeDistributed(Dense(vocab_size,
                                        name = 'timedistributed_1', kernel_regularizer=regularizer, 
                                        bias_regularizer=regularizer)))

        model.add(Activation('softmax', name='activation_1'))
        """
        """


        #MODELO No 2
        """
        #give_prob = 0.5
        model.add(LSTM(rnn_size, return_sequences=True))
        model.add(AttentionDecoder(rnn_size, vocab_size)
        #model.add(AttentionDecoder(rnn_size, vocab_size,return_probabilities=give_prob))
        """

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #K.set_value(model.optimizer.lr,np.float32(LR))

        return model

    '''def training(self, model, EPOCH, X, Y):
        for k in range(EPOCH + 1):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            for i in range(0, len(x), 2000):
                if i + 2000 >= len(x):
                    i_end = len(x)
                else:
                    i_end = i + 2000

                y_sequence = '''

   
def str_shape(x):
    return 'x'.join(map(str,x.shape))

def inspect_model(model):
    """Print the structure of Keras `model`."""
    for i, l in enumerate(model.layers):
        print(i, 'cls={} name={}'.format(type(l).__name__, l.name))
        weights = l.get_weights()
        print_str = ''
        for weight in weights:
            print_str += str_shape(weight) + ' '
        print(print_str)
        print()

def test_gen(gen, idx2word, n=5):
    Xtr,Ytr = next(gen)
    for i in range(n):
        assert Xtr[i,maxlend] == eos
        x = Xtr[i,:maxlend]
        y = Xtr[i,maxlend:]
        yy = Ytr[i,:]
        yy = np.where(yy)[1]
        prt('L',yy, idx2word)
        prt('H',y, idx2word)
        if maxlend:
            prt('D',x, idx2word)

def prt(label, X, idx2word):
    print (label+':')
    for w in X:
        sys.stdout.write(idx2word[w] + " ")
    print()


if __name__ == "__main__":
    vocH = vocabHandler()

    embedding, idx2word, word2idx, X, Y, glove_idx2idx = vocH.parse_dataset()
    print("EMBEDDING CRIADO")
    nn = neuralNetwork()

    vocab_size, embedding_size = embedding.shape
    oov0 = vocab_size - nb_unknown_words

    empty  = 0
    eos  =  1
    idx2word[empty] = '_'
    idx2word[eos] = '~'

    X_train, X_test, Y_train, Y_test = nn.split_sets(X, Y)
    batch_size = 10

    print ('dimension of embedding space for words',embedding_size)
    print ('vocabulary size', vocab_size, 'the last %d words can be used as place holders for unknown/oov words'%nb_unknown_words)
    print ('total number of different words',len(idx2word), len(word2idx))
    print ('number of words outside vocabulary which we can substitue using glove similarity', len(glove_idx2idx))
    print ('number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',len(idx2word)-vocab_size-len(glove_idx2idx))
    
    model = nn.create_model(embedding, idx2word, word2idx)
    model.summary()
    print("MODELO CRIADO")
    #Treinamento

    dh = dataHandler(oov0, glove_idx2idx, maxlend, maxlenh, vocab_size, nb_unknown_words)
    
    """
    traingen = dh.gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
    valgen = dh.gen(X_test, Y_test,  nb_batches=nb_val_samples//batch_size, batch_size=batch_size)

    for iteration in range(5):
        print('Iteration', iteration)
        h = model.fit_generator(traingen, validation_data=valgen, steps_per_epoch=nb_train_samples,
                            epochs=1, validation_steps=nb_val_samples)

        model.save_weights('modelweights.hdf5', overwrite=True)

        #for k,v in h.history.items():
        #    history[k] = history.get(k,[]) + v
            

        #with open('data/%s.history.pkl'%FN,'wb') as fp:
        #    pickle.dump(history,fp,-1)
        #
        #gensamples(batch_size=batch_size)
    """
    #TESTE

    model.load_weights("modelweights_short_run.hdf5")

    random.seed(123456789+seed)
    for t in range(20):
        xds =[]
        xhs =[]

        xd = X_test[t]
        s = random.randint(min(maxlend,len(xd)), max(maxlend,len(xd)))
        xds.append(xd[:s])
        
        xh = Y_test[t]
        s = random.randint(min(maxlenh,len(xh)), max(maxlenh,len(xh)))
        xhs.append(xh[:s])


        x, y = dh.conv_seq_labels(xds, xhs)

        sequences = []
        predictions = np.argmax(model.predict(x, verbose=0, batch_size=100), axis=2)

        for pred in predictions:
            valids = [idx2word[index] for index in pred if index > 0]
            valids2 = [print(index) for index in pred if index > 0]
            sequence = ' '.join(valids)
            sequences.append(sequence)




        xAux = ""
        yAux = ""

        for c in Y_test[t]:
            yAux += idx2word[c] + " "

        for c in X_test[t]:
            xAux += idx2word[c] + " "

        print("Original Header: " + yAux)
        print("Artificial Header: " + sequences[0])
        #print("Text: " + xAux)
        print("\n\n")




    #test_gen(dh.gen(X_train, Y_train, nflips=6, model=model, batch_size=batch_size), idx2word)

    #inspect_model(model)