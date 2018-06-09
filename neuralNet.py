import keras
import matplotlib.pyplot as plt
from vocabulary_embedding import vocabHandler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
from sklearn.cross_validation import train_test_split
import keras.backend as K
import numpy as np

seed = 42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size=64
nflips=10
maxlend=25 # 0 - if we dont want to use description at all
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 512 # must be same as 160330-word-gen
rnn_layers = 3  # match FN1
batch_norm=False
activation_rnn_size = 40 if maxlend else 0

def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    
    # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    # make sure we dont use description words that are masked out
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
    
    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))


class SimpleContext(Lambda):
    def __init__(self,**kwargs):
        super(SimpleContext, self).__init__(simple_context,**kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlend:]
    
    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)

class neuralNetwork():
    def split_sets(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)
        #len(X_train), len(Y_train), len(X_test), len(Y_test)
        return X_train, X_test, Y_train, Y_test

    def create_model(self, embedding, idx2word, word2idx):
        #activation_rnn_size = 40 if maxlend else 0
        random.seed(seed)
        np.random.seed(seed)

        regularizer = l2(weight_decay) if weight_decay else None

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_size,
                            input_length=maxlen,
                            W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True,
                            name='embedding_1'))
        for i in range(rnn_layers):
            lstm = LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                        W_regularizer=regularizer, U_regularizer=regularizer,
                        b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
                        name='lstm_%d'%(i+1)
                          )
            model.add(lstm)
            model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))

        if activation_rnn_size:
            model.add(SimpleContext(name='simplecontext_1'))
        model.add(TimeDistributed(Dense(vocab_size,
                                        W_regularizer=regularizer, b_regularizer=regularizer,
                                        name = 'timedistributed_1')))
        model.add(Activation('softmax', name='activation_1'))

        return model





if __name__ == "__main__":
    vocH = vocabHandler()

    embedding, idx2word, word2idx, X, Y = vocH.parse_dataset()

    nn = neuralNetwork()
    X_train, X_test, Y_train, Y_test = nn.split_sets(X, Y)
    model = nn.create_model(embedding, idx2word, word2idx)