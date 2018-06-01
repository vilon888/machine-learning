import numpy as np
import matplotlib.pyplot as plt
from testCases import  *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)

X, Y = load_planar_dataset()
#
# plt.scatter(X[0, :], X[1, :], c=Y.reshape(Y.shape[1]), s=40, cmap=plt.cm.Spectral)
#
#
# ### START CODE HERE ### (≈ 3 lines of code)
# shape_X = X.shape
# shape_Y = Y.shape
# m = X.shape[1]  # training set size
# ### END CODE HERE ###
#
# print ('The shape of X is: ' + str(shape_X))
# print ('The shape of Y is: ' + str(shape_Y))
# print ('I have m = %d training examples!' % (m))
#
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.reshape(Y.shape[1]))
#
# # Plot the decision boundary for logistic regression
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# plt.show()
#
# # Print accuracy
# LR_predictions = clf.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#        '% ' + "(percentage of correctly labelled datapoints)")
#

# construct neural network

# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """

    n_x = X.shape[0]

    n_h = 4  # with 1 hidden layer
    n_y = Y.shape[0]

    return n_x, n_h, n_y


# X_assess, Y_assess = layer_sizes_test_case()
# (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
# print("The size of the input layer is: n_x = " + str(n_x))
# print("The size of the hidden layer is: n_h = " + str(n_h))
# print("The size of the output layer is: n_y = " + str(n_y))


# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parameters


# n_x, n_h, n_y = initialize_parameters_test_case()
#
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.matmul(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache

# X_assess, parameters = forward_propagation_test_case()
#
# A2, cache = forward_propagation(X_assess, parameters)
#
# # Note: we use the mean here just to make sure that your output matches ours.
# print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

# GRADED FUNCTION: compute_cost


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = A2.shape[1]
    costs = (- 1/m) * np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2)))

    return costs


# A2, Y_assess, parameters = compute_cost_test_case()
#
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# GRADED FUNCTION: backward_propagation

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """

    m = X.shape[1]

    dZ2 = cache['A2'] - Y
    dW2 = (1/m) * np.dot(dZ2, cache['A1'].T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.matmul(parameters['W2'].T, dZ2), (1-np.power(cache['A1'], 2)))
    dW1 = (1/m) * np.matmul(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return grads

# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))


# GRADED FUNCTION: update_parameters
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    W1 = parameters['W1'] - learning_rate * grads['dW1']
    W2 = parameters['W2'] - learning_rate * grads['dW2']
    b1 = parameters['b1'] - learning_rate * grads['db1']
    b2 = parameters['b2'] - learning_rate * grads['db2']

    return {
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2
    }


# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations=10000, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs -- cost for iterations
    """

    np.random.seed(3)
    n_x, _, n_y = layer_sizes(X, Y)

    parameters = initialize_parameters(n_x, n_h, n_y)

    costs = []
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)

        if i % 1000 == 0:
            costs.append(cost)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {} is : {}".format(i, cost))

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate)

    return parameters, costs


# X_assess, Y_assess = nn_model_test_case()
#
# parameters, costs = nn_model(X_assess, Y_assess, 4, num_iterations=10000, learning_rate=1.2, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# GRADED FUNCTION: predict
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    A2, _ = forward_propagation(X, parameters)
    predictions = (A2 > 0.5).astype(np.int)
    return predictions


# parameters, X_assess = predict_test_case()
#
# predictions = predict(parameters, X_assess)
# print("predictions mean = " + str(np.mean(predictions)))


# # Build a model with a n_h-dimensional hidden layer
# parameters, _ = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
#
# # Plot the decision boundary
# plt.scatter(X[0, :], X[1, :], c=Y.reshape(Y.shape[1]), s=40, cmap=plt.cm.Spectral)
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))
#
# plt.show()


# This may take about 2 minutes to run

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):

    parameters, costs = nn_model(X, Y, n_h, learning_rate=0.8, num_iterations=5000)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    plt.scatter(X[0, :], X[1, :], c=Y.reshape(Y.shape[1]), s=40, cmap=plt.cm.Spectral)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

plt.show()
