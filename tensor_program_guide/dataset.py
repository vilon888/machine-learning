import tensorflow as tf

sess = tf.Session()

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sum = 0
while True:
    try:
        print(sess.run(next_element))
        print('--------------------------------------------')
        sum += 1
    except tf.errors.OutOfRangeError:
        print('the End!')
        print(sum)
        break

#
#
# print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
# print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
#                                #      [5, 5, 5, 5, 5, 0, 0],
#                                #      [6, 6, 6, 6, 6, 6, 0],
#                                #      [7, 7, 7, 7, 7, 7, 7]]