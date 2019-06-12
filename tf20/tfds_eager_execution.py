import os
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

x = tf.random.uniform([16,10], -10, 0, tf.int64)

DS = tf.data.Dataset.from_tensor_slices((x))


def mapfunc(ex, con):
    # import pdb; pdb.set_trace()
    new_ex = ex + con
    print(new_ex) 
    return new_ex

# DS = DS.map(lambda x: mapfunc(x, [7]))
DS = DS.map(
lambda x: tf.py_function(
  mapfunc,
  [x, tf.constant(7, dtype=tf.int64)], tf.int64)
)
for i in DS:
    ...
    # print(i)