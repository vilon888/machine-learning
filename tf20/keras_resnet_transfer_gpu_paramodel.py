import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

# from tensorflow.python.framework import set_gpu_per_process_memory_growth
# from tensorflow.python.framework import gpu
# physical_devices = config.experimental.list_physical_devices('GPU')
from tensorflow.python.framework import config
config.set_gpu_per_process_memory_growth(True)
import tensorflow.python
# for gpu in physical_devices:
#     config.experimental.set_memory_growth(device=gpu, enable=True)

from PIL import Image

# import onnxmltools
from MyDataset import MyDataset


def get_open_shelf_dataset():
    dl_config = tfds.download.DownloadConfig(
        manual_dir='/home/vilon_tao/tensorflow_datasets/downloads/manual',
        download_mode=tfds.GenerateMode.REUSE_DATASET_IF_EXISTS
    )

    train_ds, test_ds, val_ds = tfds.load(
        name='my_dataset',
        data_dir='/home/taolongming/tensorflow_datasets',
        split=["train", "test", "val"],
        download=False,
        # as_supervised=True,
        builder_kwargs=dict(dataset_name='open.shelf.classfication'),
        download_and_prepare_kwargs=dict(download_config=dl_config),
    )

    return train_ds, test_ds, val_ds




class Embeddings(tf.keras.layers.Layer):
    def __init__(self, embedding_size, **kwargs):
        self.embedding_size = embedding_size
        self.dense = tf.keras.layers.Dense(self.embedding_size, name='fc_{}'.format(self.embedding_size))
        super(Embeddings, self).__init__(**kwargs)

    def call(self, inputs):
        return self.dense(inputs)

    def get_config(self):
        config = super(Embeddings, self).get_config()
        config.update({'embedding_size': self.embedding_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Classification(tf.keras.layers.Layer):
    def __init__(self, class_num, **kwargs):
        self.class_num = class_num
        super(Classification, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.class_num))
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)

    def call(self, inputs):
        kernel_norm = tf.norm(self.kernel, ord=2, axis=-1, keepdims=True)
        kernel_norm = tf.divide(self.kernel, kernel_norm)
        inputs_norm = tf.norm(inputs, ord=2, axis=1, keepdims=True)
        inputs = tf.divide(inputs, inputs_norm)
        prediction = tf.matmul(inputs, kernel_norm)
        return prediction

    def get_config(self):
        config = super(Classification, self).get_config()
        config.update({'class_num': self.class_num})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_open_shelf_resnet(input_shape=[96, 96, 3], class_num=7, embedding_size=512):
    model = tf.keras.applications.ResNet50(include_top=False,
                                           weights='imagenet',
                                           input_shape=input_shape)
    base_layer = model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(base_layer)
    embedding = Embeddings(embedding_size)(x)
    prediction = Classification(class_num)(embedding)

    open_shelf_model = tf.keras.Model(inputs=model.input,
                                      # outputs=[embedding, prediction],
                                      outputs=prediction,
                                      name='open_shelf_resnet_model')

    return open_shelf_model

#
def resize(item):
    H = tf.cast(item['height'], dtype=tf.float32)
    W = tf.cast(item['width'], dtype=tf.float32)
    scale = tf.random.uniform([], minval=0.8, maxval=1, dtype=tf.float32)

    new_H = tf.cast(H * scale, dtype=tf.int32)
    new_W = tf.cast(W * scale, dtype=tf.int32)
    item['image'] = tf.image.random_crop(item['image'], size=(new_H, new_W, 3))

    item['image'] = tf.image.resize(item['image'], (96, 96))
    item['image'] = tf.cast(item['image'] / 255, dtype=tf.float32)
    return item['image'], item['label']

# def resize(image, label):
#     # H = tf.cast(item['height'], dtype=tf.float32)
#     # W = tf.cast(item['width'], dtype=tf.float32)
#     # scale = tf.random.uniform([], minval=0.8, maxval=1, dtype=tf.float32)
#     #
#     # new_H = tf.cast(H * scale, dtype=tf.int32)
#     # new_W = tf.cast(W * scale, dtype=tf.int32)
#     # item['image'] = tf.image.random_crop(item['image'], size=(new_H, new_W, 3))
#     #
#     # item['image'] = tf.image.resize(item['image'], (96, 96))
#     image = tf.image.resize(image, (96, 96))
#     image = image / 255
#     return image, label


def resize_val_test(item):
    item['image'] = tf.image.resize(item['image'], (96, 96))
    item['image'] = tf.cast(item['image'] / 255, dtype=tf.float32)
    return item['image'], item['label']


def get_tuple(item):
    return item['image'], item['label']  # (item['label'], item['label'])


def get_imgs(item):
    return item['image']


#
# def test_model():
#     # new_model = tf.keras.experimental.load_from_saved_model('models/open_shelf_resnet_model')
#     new_model = tf.keras.models.load_model('models/open_shelf_resnet_model_05.h5',
#                                            custom_objects={'Classification': Classification,
#                                                            'Embeddings': Embeddings,
#                                                            'am_softmax_loss': am_softmax_loss}
#                                            )
#
#     sum = 0
#     correct = 0
#     wrong = 0
#     for images, labels in test_ds_model:
#         results = new_model.predict(images)
#         sum += 1
#         if np.argmax(results[1]) ==labels.numpy():
#             correct += 1
#         else:
#             wrong +=1
#
#     print('correct ratio: {}\n wrong ratio: {}'.format(correct/sum, wrong/sum))


def train_save_model():
    # mirrored_strategy = tf.distribute.MirroredStrategy()

    # with mirrored_strategy.scope():
    #     def am_softmax_loss(label, prediction, m=0.35, s=30):
    #         cos_theta = tf.clip_by_value(prediction, -1, 1)
    #         phi = cos_theta - m
    #
    #         label1 = tf.cast(tf.keras.backend.flatten(label), 'int64')
    #
    #         label_onehot = tf.one_hot(label1, 7)
    #         adjust_theta = s * tf.where(tf.equal(label_onehot, 1), phi, cos_theta)
    #         loss = tf.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    #         return loss(label_onehot, adjust_theta)

    def am_softmax_loss(label, prediction, m=0.35, s=30):
        cos_theta = tf.clip_by_value(prediction, -1, 1)
        phi = cos_theta - m

        label1 = tf.cast(tf.keras.backend.flatten(label), 'int64')

        label_onehot = tf.one_hot(label1, 7)
        adjust_theta = s * tf.where(tf.equal(label_onehot, 1), phi, cos_theta)
        loss = tf.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        return loss(label_onehot, adjust_theta)

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    # BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 3

    train_ds, test_ds, val_ds = get_open_shelf_dataset()

    # for i in train_ds:
    #     image, label = i
    #     print(i)
    train_ds = train_ds.map(resize).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    val_ds = val_ds.map(resize_val_test).batch(BATCH_SIZE, drop_remainder=True)
    test_ds_eval = test_ds.map(resize_val_test).batch(BATCH_SIZE, drop_remainder=True)
    test_ds_predict = test_ds.map(resize_val_test).batch(BATCH_SIZE, drop_remainder=True)

    test_ds_model = test_ds.map(resize_val_test).batch(BATCH_SIZE, drop_remainder=True)

    # with mirrored_strategy.scope():
    #
    #
    #     model = get_open_shelf_resnet()
    #
    #     # model.summary()
    #
    #     filepath = os.path.join('./models', model.name + "_{epoch:02d}.h5")
    #
    #     callbacks = [
    #         tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='batch'),
    #         tf.keras.callbacks.ModelCheckpoint(filepath)
    #     ]
    #     optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4)
    #     model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4),
    #                   # loss=[None, am_softmax_loss],
    #                   loss=am_softmax_loss,
    #                   # metrics=['accuracy'],
    #                   # metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)],
    #                   metrics=['sparse_categorical_accuracy']
    #
    #                   )
    # with tf.device('/cpu:0'):
    model = get_open_shelf_resnet()

    parallel_model = tf.keras.utils.multi_gpu_model(model, 3, cpu_relocation=True)

    # model.summary()

    filepath = os.path.join('./models', model.name + "_{epoch:02d}.h5")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='batch'),
        tf.keras.callbacks.ModelCheckpoint(filepath)
    ]
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4)
    parallel_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4),
                  # loss=[None, am_softmax_loss],
                  loss=am_softmax_loss,
                  # metrics=['accuracy'],
                  # metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)],
                  metrics=['sparse_categorical_accuracy']

                  )

        # for epoch in range(3):
        #     print('Start of epoch %d' % (epoch,))
        #     for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
        #         # Open a GradientTape to record the operations run
        #         # during the forward pass, which enables autodifferentiation.
        #         with tf.GradientTape() as tape:
        #
        #             # Run the forward pass of the layer.
        #             # The operations that the layer applies
        #             # to its inputs are going to be recorded
        #             # on the GradientTape.
        #             logits = model(x_batch_train)  # Logits for this minibatch
        #
        #             # Compute the loss value for this minibatch.
        #             loss_value = am_softmax_loss(y_batch_train, logits)
        #
        #         # Use the gradient tape to automatically retrieve
        #         # the gradients of the trainable variables with respect to the loss.
        #         grads = tape.gradient(loss_value, model.trainable_variables)
        #
        #         # Run one step of gradient descent by updating
        #         # the value of the variables to minimize the loss.
        #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
        #
        #         # Log every 200 batches.
        #         if step % 200 == 0:
        #             print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
        #             print('Seen so far: %s samples' % ((step + 1) * 48))
    parallel_model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)
    # parallel_model.fit(train_ds, epochs=5,  callbacks=callbacks, steps_per_epoch=1000)
    # model.fit(train_ds, epochs=5,  callbacks=callbacks)

    eval_results = parallel_model.evaluate(test_ds_eval)

    print(eval_results)

    results = parallel_model.predict(test_ds_predict)

    print(results)

    model.save(os.path.join('./models', model.name + '.h5'))

    tf.keras.experimental.export_saved_model(model, os.path.join('./models/' + model.name))

    tf.keras.utils.plot_model(model, os.path.join('./models', model.name + '.png'), show_shapes=True)


# onnx_model = onnxmltools.convert_keras(model)


# for i in train_ds:
#     sinogram = tf.image.encode_jpeg(i['image'], quality=100)
#     writer = tf.io.write_file('test_sinogram.jpg', sinogram)
#     print('saved image')


# train_ds = train_ds.map(resize).shuffle(1000).batch(64)
# val_ds = val_ds.map(resize_val_test).batch(64)
# test_ds_eval = test_ds.map(resize_val_test).batch(64)
# test_ds_predict = test_ds.map(resize_val_test).batch(64)
#
# test_ds_model = test_ds.map(resize_val_test).batch(1)


if __name__ == '__main__':
    train_save_model()
    # test_model()
