import os

import tensorflow as tf
from tensorflow import keras

import pathlib


(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.


def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(28 * 28, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


model = create_model()
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# load trained weights
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

model.save_weights('./training_3/my_checkpoint')

#
# checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# cp_callback = keras.callbacks.ModelCheckpoint(
#     checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     period=5)
#
# model = create_model()
# model.fit(train_images,
#           train_labels,
#           epochs=50,
#           callbacks=[cp_callback],
#           validation_data=(test_images, test_labels),
#           verbose=0)
#
# checkpoints = pathlib.Path(checkpoint_dir).glob('*.index')
# checkpoints = sorted(checkpoints, key=lambda cp: cp.stat().st_mtime)
# checkpoints = [cp.with_suffix('') for cp in checkpoints]
# latest = str(checkpoints[-1])
#
# print(checkpoints)
#
# model = create_model()
# model.load_weights(latest)
# loss, acc = model.evaluate(test_images, test_labels)
# print('Restored model , accuracy: {:5.2f}%'.format(acc*100))
#
#
# model.save_weights('./training_3')
# model = create_model()
# model.load_weights('./training_3')
# loss,acc = model.evaluate(test_images, test_labels)
# print("Restored model from save_weights call, accuracy: {:5.2f}%".format(100*acc))
#
#

# # Save entire model
# model = create_model()
# model.fit(train_images, train_labels, epochs=5)
# model.save('training_4/my_model.h5')
#
# new_model = keras.models.load_model('training_4/my_model.h5')
# new_model.summary()
# loss, acc = new_model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))


















