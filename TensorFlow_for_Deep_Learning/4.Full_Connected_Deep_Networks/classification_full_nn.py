import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

np.random.seed(1)
tf.set_random_seed(1)

# Generate synthetic data

N = 100
w_true = 5
b_true = 2
noise_scale = .1

x_zeros = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=.1 * np.eye(2), size=(N//2,))
y_zeros = np.zeros((N//2, 1), dtype=np.float32)

x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=.1 * np.eye(2), size=(N//2, ))
y_ones = np.ones((N//2, 1), dtype=np.float32)

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])

# plot scatter
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("FCNet Classification Data")

# Plot Zeros
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color="blue")
plt.scatter(x_ones[:, 0], x_ones[:, 1], color="red")
plt.show()


# tf graph

d = 2
n_hidden = 15

with tf.name_scope('placeholder'):
    x = tf.placeholder(tf.float32, (N, d))
    y = tf.placeholder(tf.float32, (N, 1))

with tf.name_scope('layer-1'):
    W = tf.Variable(tf.random_normal((n_hidden, d)))
    b = tf.Variable(tf.random_normal((1, n_hidden)))
    A1 = tf.nn.relu(tf.matmul(x, tf.transpose(W)) + b)

with tf.name_scope('output'):
    W = tf.Variable(tf.random_normal((1, n_hidden)))
    b = tf.Variable(tf.random_normal((1, 1)))
    Z2 = tf.matmul(A1, tf.transpose(W))

    y_prob = tf.sigmoid(Z2)
    y_pred = tf.round(y_prob)

with tf.name_scope('loss'):
    entroy = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2, labels=y_np)
    l = tf.reduce_sum(entroy)

with tf.name_scope('optim'):
    train_op = tf.train.AdamOptimizer(.001).minimize(l)

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', l)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/tf_deep/ch4/classification_full_nn', tf.get_default_graph())

n_steps = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(n_steps):
        _, summary, loss = sess.run([train_op, merged, l],
                                    feed_dict={x: x_np, y: y_np})
        train_writer.add_summary(summary, i)
        print('After step {}, the loss is: {}'.format(i, loss))

    y_pred_np = sess.run(y_pred, feed_dict={x: x_np})


score = accuracy_score(y_np, y_pred_np)
print("Classification Accuracy: %f" % score)