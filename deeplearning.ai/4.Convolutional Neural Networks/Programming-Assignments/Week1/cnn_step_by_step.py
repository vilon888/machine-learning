import numpy as np
import matplotlib.pyplot as plt
import h5py


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                   'constant', constant_values=0)

    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    tmp = np.multiply(W, a_slice_prev)
    Z = np.sum(tmp) + float(b)
    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    n_H = (n_H_prev - f + 2*pad) // stride + 1
    n_W = (n_W_prev - f + 2*pad) // stride + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start  = h * stride
                    vert_end    = vert_start + f
                    horiz_start = w * stride
                    horiz_end   = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end,
                                   horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev,
                                                     W[..., c], b[..., c])

    cache = A_prev, W, b, hparameters

    return Z, cache


def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f = hparameters['f']
    stride = hparameters['stride']
    n_H = (n_H_prev - f) // stride + 1
    n_W = (n_W_prev - f) // stride + 1

    A = np.zeros((m, n_H, n_W, n_C_prev))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C_prev):
                    v_start = h * stride
                    v_end   = v_start + f
                    h_start = w * stride
                    h_end   = h_start + f

                    a_prev_slice = A_prev[i, v_start:v_end,
                                   h_start:h_end, c]
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.average((a_prev_slice))

    cache = A_prev, hparameters

    return A, cache


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    A_prev, W, b, hparameters = cache
    m, n_H, n_W, n_C = dZ.shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_pre_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride
                    v_end = v_start + f
                    h_start = w * stride
                    h_end = h_start + f

                    a_slice = a_pre_pad[v_start:v_end, h_start:h_end, :]

                    da_prev_pad[v_start:v_end, h_start:h_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, db


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = x == np.max(x)
    return mask


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape

    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    n_H, n_W = shape

    avg = dz / (n_H * n_W)

    a = np.ones(shape) * avg

    return a


def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    A_prev, hparameters = cache
    stride = hparameters['stride']
    f = hparameters['f']
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride
                    v_end = v_start + f
                    h_start = w * stride
                    h_end = h_start + f

                    if mode == 'max':
                        a_prev_slice = a_prev[v_start:v_end, h_start:h_end, c]
                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i, v_start:v_end, h_start:h_end, c] += \
                        np.multiply(mask, dA[i, h, w, c])
                    elif mode == 'average':
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += \
                        distribute_value(da, shape)

    return dA_prev

# # test pad
# np.random.seed(1)
# x = np.random.randn(4, 3, 3, 2)
# x_pad = zero_pad(x, 2)
# print("x.shape =", x.shape)
# print("x_pad.shape =", x_pad.shape)
# print("x[1,1] =", x[1, 1])
# print("x_pad[1,1] =", x_pad[1, 1])
#
# fig, axarr = plt.subplots(1, 2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0, :, :, 0])
# axarr[1].set_title('x_pad')
# axarr[1].imshow(x_pad[0, :, :, 0])
# plt.show()

# np.random.seed(1)
# a_slice_prev = np.random.randn(4, 4, 3)
# W = np.random.randn(4, 4, 3)
# b = np.random.randn(1, 1, 1)
#
# Z = conv_single_step(a_slice_prev, W, b)
# print("Z =", Z)

np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad": 2,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("Z's mean =", np.mean(Z))
# print("Z[3,2,1] =", Z[3, 2, 1])
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])


# np.random.seed(1)
# A_prev = np.random.randn(2, 4, 4, 3)
# hparameters = {"stride" : 2, "f": 3}
#
# A, cache = pool_forward(A_prev, hparameters)
# print("mode = max")
# print("A =", A)
# print()
# A, cache = pool_forward(A_prev, hparameters, mode = "average")
# print("mode = average")
# print("A =", A)

np.random.seed(1)
dA, dW, db = conv_backward(Z, cache_conv)
print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))


np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])
