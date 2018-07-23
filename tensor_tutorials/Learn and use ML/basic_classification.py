import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print('train_images shape is: {}'.format(train_images.shape))
# print('train_labels shape is: {}'.format(train_labels.shape))
# print('test_images shape is: {}'.format(test_images.shape))
# print('test_labels shape is: {}'.format(test_labels.shape))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.gca().grid(False)
# plt.show()

train_images = train_images / 255.
test_images = test_images / 255.

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid('off')
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
#
# plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test loss is: {}, Test accuracy is: {}', test_loss, test_accuracy)

predictions = model.predict(test_images)
print(predictions[0])

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel('{} ({})'.format(class_names[predicted_label], class_names[true_label]), color=color)
plt.show()

image = test_images[0]
image = np.expand_dims(image, axis=0)

single_pred = model.predict(image)
print(class_names[np.argmax(single_pred[0])])

