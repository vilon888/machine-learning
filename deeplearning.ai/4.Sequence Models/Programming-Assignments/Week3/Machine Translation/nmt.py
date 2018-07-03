from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor(concat)

    alphas = activator(e)
    context = dotor([alphas, a])

    return context


def create_model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s, ), name='s0')
    c0 = Input(shape=(n_s, ), name='c0')
    s = s0
    c = c0
    outputs = []
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    for t in range(Ty):
        context = one_step_attention(a, s)

        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        out = output_layer(s)

        outputs.append(out)

    model = Model([X, s0, c0], outputs)

    return model


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor = Dense(1, activation='relu')
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)

n_a = 64
n_s = 128
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation=softmax)


model = create_model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))

# model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

model.load_weights('models/model.h5')

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x : to_categorical(x, num_classes=len(human_vocab)), source)))
    source = np.expand_dims(source, axis=0)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print('source:', example)
    print('output:', ''.join(output))


model.summary()

attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday April 08 1993", num = 6, n_s = 128)































