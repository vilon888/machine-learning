import numpy as np
from utils import *
import random


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''

    for key in gradients.keys():
        np.clip(gradients[key], -maxValue, maxValue, out=gradients[key])

    return gradients


def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    indices = []
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a, _ = Waa.shape

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    idx = -1

    counter = 0
    eos = char_to_ix['\n']

    while idx != eos and counter != 50:
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        y_pred = softmax(np.dot(Wya, a) + by)

        np.random.seed(counter + seed)
        idx = np.random.choice(list(range(vocab_size)), p=y_pred.ravel())
        indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        a_prev = a

        seed += 1
        counter += 1

    if counter == 50:
        indices.append(char_to_ix['\n'])

    return indices


def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    gradients, a = rnn_backward(X, Y, parameters, cache)

    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]


def model(data, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """
    n_x = n_y = vocab_size

    parameters = initialize_parameters(n_a, n_x, n_y)

    loss = get_initial_loss(vocab_size, dino_names)

    with open('dinos.txt') as f:
        examples = f.readlines()

    examples = [x.lower().strip() for x in examples]
    np.random.seed(0)
    np.random.shuffle(examples)

    a_prev = np.zeros((n_a, 1))

    for j in range(num_iterations):
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix['\n']]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)

        loss = smooth(loss, curr_loss)

        if j % 2000 == 0:
            print('Iteration: {}, Loss: {} \n'.format(j, loss))

            seed = 0
            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                seed += 1
            print('\n')

    return parameters


# # testing clip
# np.random.seed(3)
# dWax = np.random.randn(5,3)*10
# dWaa = np.random.randn(5,5)*10
# dWya = np.random.randn(2,5)*10
# db = np.random.randn(5,1)*10
# dby = np.random.randn(2,1)*10
# gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
# gradients = clip(gradients, 10)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
# print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"dby\"][1] =", gradients["dby"][1])

data = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))

print(sorted(chars))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}

# # testing sample
# np.random.seed(2)
# _, n_a = 20, 100
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
#
#
# indices = sample(parameters, char_to_ix, 0)
# print("Sampling:")
# print(len(indices))
# print("list of sampled indices:", indices, '\n')
# print("list of sampled characters:", [ix_to_char[i] for i in indices])


# # testing optimize
# np.random.seed(1)
# vocab_size, n_a = 27, 100
# a_prev = np.random.randn(n_a, 1)
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
# X = [12,3,5,11,22,3]
# Y = [4,14,11,22,25, 26]
#
# loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
# print("Loss =", loss)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
# print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"dby\"][1] =", gradients["dby"][1])
# print("a_last[4] =", a_last[4])

# training model
parameters = model(data, ix_to_char, char_to_ix)







