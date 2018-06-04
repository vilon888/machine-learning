import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score


# Load the Tox21 dataset

_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Cause we have a fewer positive examples, so adding different weights for these kinds of examples
# But here we remove extra weight temporarily
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]


# generate tf graph
d = 1024
n_hidden = 50
learning_rate = .001
n_epochs = 10
batch_size = 100

with tf.name_scope('placeholder'):
    x = tf.placeholder(tf.float32, (None, d))
    y = tf.placeholder(tf.float32, (None, ))

with tf.name_scope('hidden-layer'):
    W = tf.Variable(tf.random_normal((d, n_hidden)))
    b = tf.Variable(tf.random_normal((n_hidden, )))
    x_hidden = tf.nn.relu(tf.matmul(x, W) + b)


with tf.name_scope('output'):
    W = tf.Variable(tf.random_normal((n_hidden, 1)))
    b = tf.Variable(tf.float32, (1,))
    y_logit = tf.matmul(x_hidden, W) + b

    # sigmoid as the activition function
    y_one_prob = tf.sigmoid(y_logit)

    y_pred = tf.round(y_one_prob)


with tf.name_scope('loss'):
    y_expand = tf.expand_dims(y, 1)
    entroy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
    l = tf.reduce_sum(entroy)

with tf.name_scope('optim'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)


with tf.name_scope('summaries'):
    tf.summary.scalar('loss', l)
    merged = tf.summary.merge_all()


train_writer = tf.summary.FileWriter('/tmp/tf_deep/ch3/tox21_full_connect', tf.get_default_graph())

N = train_X.shape[0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(n_epochs):
        pos = 0
        while pos < N:
            batch_X = train_X[pos:pos + batch_size]
            batch_Y = train_y[pos:pos + batch_size]
            _, summary, loss = sess.run([train_op, merged, l],
                                        feed_dict= {x: batch_X, y: batch_Y})
            print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
            train_writer.add_summary(summary, step)

            step += 1
            pos += batch_size

    #make prediction
    valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X})

score = accuracy_score(valid_y, valid_y_pred)
print("Unweighted Classification Accuracy: %f" % score)

weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
print("Weighted Classification Accuracy: %f" % weighted_score)


