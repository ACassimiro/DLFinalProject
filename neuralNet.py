import keras
import matplotlib.pyplot as plt
from vocabulary_embedding import vocabHandler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.engine.topology import Layer
from keras.layers import SpatialDropout1D
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
from keras import initializers
from keras import regularizers
from keras import constraints
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam, RMSprop
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
nb_train_samples = 30000
nb_val_samples = 3000


class SimpleContext(Lambda):
    """Class to implement `simple_context` method as a Keras layer."""

    def __init__(self, fn, rnn_size, **kwargs):
        """Initialize SimpleContext."""
        self.rnn_size = rnn_size
        super(SimpleContext, self).__init__(fn, **kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        """Compute mask of maxlend."""
        return input_mask[:, maxlend:]

    def get_output_shape_for(self, input_shape):
        """Get output shape for a given `input_shape`."""
        nb_samples = input_shape[0]
        n = 2 * (self.rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)


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
                            input_length=maxlen,
                            W_regularizer=regularizer, weights=[embedding], mask_zero=True,
                            name='embedding_1'))
        model.add(SpatialDropout1D(p=p_emb))
        for i in range(rnn_layers):
            lstm = LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                        W_regularizer=regularizer, U_regularizer=regularizer,
                        b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
                        name='lstm_%d'%(i+1)
                          )
            model.add(lstm)
            model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))


        def simple_context(X, mask, n=activation_rnn_size):
            """Reduce the input just to its headline part (second half).
            For each word in this part it concatenate the output of the previous layer (RNN)
            with a weighted average of the outputs of the description part.
            In this only the last `rnn_size - activation_rnn_size` are used from each output.
            The first `activation_rnn_size` output is used to computer the weights for the averaging.
            """
            desc, head = X[:, :maxlend, :], X[:, maxlend:, :]
            head_activations, head_words = head[:, :, :n], head[:, :, n:]
            desc_activations, desc_words = desc[:, :, :n], desc[:, :, n:]

            # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
            # activation for every head word and every desc word
            activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2, 2))
            # make sure we dont use description words that are masked out

            print()
            print(activation_energies)
            print(type(K.cast(mask[:, :maxlend], 'float32')))
            print(1. - K.cast(mask[:, :maxlend], 'float32'))
            print(K.cast(mask[:, :maxlend], 'float32'))
            print(K.expand_dims(1. - K.cast(mask[:, :maxlend], 'float32'), 1))
            print()

            activation_energies = activation_energies + -1e20 * K.expand_dims(
                1. - K.cast(mask[:, :maxlend], 'float32'), 1)

            # for every head word compute weights for every desc word
            activation_energies = K.reshape(activation_energies, (-1, maxlend))
            activation_weights = K.softmax(activation_energies)
            activation_weights = K.reshape(activation_weights, (-1, maxlenh, maxlend))

            # for every head word compute weighted average of desc words
            desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2, 1))
            return K.concatenate((desc_avg_word, head_words))

        if activation_rnn_size:
            simpleContext = SimpleContext(simple_context, rnn_size, name='simplecontext_1')
            model.add(simpleContext)

        print("")
        #print(attentionLayer.compute_output_shape())
        print("")

        #inspect_model(model)

        model.add(TimeDistributed(Dense(vocab_size,
                                        name = 'timedistributed_1', kernel_regularizer=regularizer, 
                                        bias_regularizer=regularizer)))

        #model.add(TimeDistributed(Dense(vocab_size,
        #                                W_regularizer=regularizer, b_regularizer=regularizer,
        #                                name = 'timedistributed_1')))

        model.add(Activation('softmax', name='activation_1'))

        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        K.set_value(model.optimizer.lr,np.float32(LR))



        return model

   
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


if __name__ == "__main__":
    vocH = vocabHandler()

    embedding, idx2word, word2idx, X, Y = vocH.parse_dataset()

    nn = neuralNetwork()
    X_train, X_test, Y_train, Y_test = nn.split_sets(X, Y)
    model = nn.create_model(embedding, idx2word, word2idx)

    #inspect_model(model)