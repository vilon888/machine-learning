import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 26
plt.imshow(train_set_x_orig[index])
plt.show()
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "' picture.")

### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
### END CODE HERE ###

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

# standardization
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    return s


# GRADED FUNCTION: initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros(shape=(dim, 1))
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.matmul(w.T, X) + b)

    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = (1 / m) * np.matmul(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    grads = {
        'dw': dw,
        'db': db
    }
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {} is : {}".format(i, cost))

    params = {
        'w': w,
        'b': b
    }

    grads = {
        'dw': dw,
        'db': db
    }

    return params, grads, costs


def predict(w, b, X):
    A = sigmoid(np.matmul(w.T, X) + b)
    prediction = (A > 0.5).astype(np.float)

    return prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=True):
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w, b = params['w'], params['b']

    prediction_train = predict(w, b, X_train)
    prediction_test = predict(w, b, X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": prediction_test,
         "Y_prediction_train": prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# with 3000 will overfiting
results = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
                print_cost=True)

print(results)

# Example of a picture that was wrongly classified.
index = 5
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
plt.show()
print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[
    np.int(results["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")

costs = np.squeeze(results['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(results["learning_rate"]))

plt.show()

#
# # try different learning rate
# learning_rates = {0.01, 0.001, 0.0001}
# models = {}
# for i in learning_rates:
#     print("learning rate is: {}".format(i))
#     models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y,
#                            1500, i, print_cost=True)
#     print('\n' + "-------------------------------------------------------" + '\n')
#
# for i in learning_rates:
#     plt.plot(models[str(i)]['costs'], label=str(i))
#
# plt.ylabel('costs')
# plt.xlabel('iterations')
#
# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()


# test your own image
my_image = "my_image2.jpg"
fname = "images/" + my_image
image = np.array(plt.imread(fname))

my_image = skimage.transform.resize(image, (num_px, num_px)).reshape((1, num_px * num_px * 3)).T

my_predicted_image = predict(results["w"], results["b"], my_image)

plt.imshow(my_image.reshape((num_px, num_px, 3)))
plt.show()

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
    int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
