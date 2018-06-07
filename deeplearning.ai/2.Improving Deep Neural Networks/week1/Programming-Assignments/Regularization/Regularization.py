import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.

    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    l2_regularization = 0
    for w in [parameters[k] for k in parameters.keys() if k.startswith('W')]:
        l2_regularization += np.sum(np.square(w))
    cost = compute_cost(A3, Y) + (1 / m * lambd / 2) * l2_regularization

    return cost


def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3 = cache
    dZ3 = A3 -Y
    dW3 = (1./m) * (np.dot(dZ3, A2.T) + lambd * W3)
    db3 = (1./m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1./m) * (np.dot(dZ2, A1.T) + lambd * W2)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1./m) * (np.dot(dZ1, X.T) + lambd * W1)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    np.random.seed(1)
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    A1 = relu(Z1)

    D1 = np.random.rand(A1.shape[0], A1.shape[1]) < keep_prob

    A1 = np.multiply(A1, D1)
    A1 = A1 / keep_prob

    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1]) < keep_prob
    A2 = np.multiply(A2, D2)
    A2 = A2 / keep_prob

    Z3 = np.dot(parameters['W3'], A2) + parameters['b3']
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, parameters['W1'], parameters['b1'],
             Z2, D2, A2, parameters['W2'], parameters['b2'],
             Z3, A3, parameters['W3'], parameters['b3'])

    return A3, cache


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1./m) * np.dot(dZ3, A2.T)
    db3 = (1./m) * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = np.multiply(dA2, D2)
    dA2 /= keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = np.multiply(dA1, D1)
    dA1 /= keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients




def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    costs = []
    grads = {}
    layer_dims = [X.shape[0], 20, 3, 1]
    parameters = initialize_parameters(layer_dims)
    for i in range(num_iterations):
        if keep_prob == 1:
            A3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            A3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        # only use one of lambd or dropout at the same time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        parameters = update_parameters(parameters, grads, learning_rate)
        if lambd == 0:
            cost = compute_cost(A3, Y)
        else:
            cost = compute_cost_with_regularization(A3, Y, parameters, lambd)
        costs.append(cost)
        if i % 1000 == 0 and print_cost:
            print('After iteration {}, cost is {}'.format(i, cost))

    # draw cost vs iteration graph
    plt.plot(costs)
    plt.xlabel('Iterations * 1000')
    plt.ylabel('cost')
    plt.title('learning_rate = ' + str(learning_rate))
    plt.show()

    return parameters


# # train model without regularization
# parameters = model(train_X, train_Y)
# print("On the training set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)
#
#
# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
#

# # test l2 computing
# A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
# print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))


# # test backward propagation with regularization
# X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
#
# grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("dW3 = "+ str(grads["dW3"]))

# # test train model with L2
# # parameters = model(train_X, train_Y, lambd=.7)
# # print ("On the train set:")
# # predictions_train = predict(train_X, train_Y, parameters)
# # print ("On the test set:")
# # predictions_test = predict(test_X, test_Y, parameters)
# #
# # plt.title("Model with L2-regularization")
# # axes = plt.gca()
# # axes.set_xlim([-0.75, 0.40])
# # axes.set_ylim([-0.75, 0.65])
# # plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# # test forward with dropout
# X_assess, parameters = forward_propagation_with_dropout_test_case()
#
# A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
# print ("A3 = " + str(A3))

# # test backward with dropout
# X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()
# gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob = 0.8)
#
# print ("dA1 = " + str(gradients["dA1"]))
# print ("dA2 = " + str(gradients["dA2"]))

# train with dropout
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)