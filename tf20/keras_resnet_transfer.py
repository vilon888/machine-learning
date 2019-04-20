import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import os


def get_open_shelf_dataset():
    dl_config = tfds.download.DownloadConfig(
        manual_dir='/home/vilon_tao/tensorflow_datasets/downloads/manual',
        download_mode=tfds.GenerateMode.REUSE_DATASET_IF_EXISTS
    )

    train_ds, test_ds, val_ds = tfds.load(
        name='image_label_folder',
        data_dir='/home/vilon_tao/tensorflow_datasets',
        split=["train", "test", "val"],
        download=True,
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
        inputs = tf.divide(inputs,inputs_norm)
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
                                      outputs=[embedding, prediction],
                                      name='open_shelf_resnet_model')

    return open_shelf_model


def resize(item):
  item['image'] = tf.image.resize(item['image'], (96, 96))
  return item

def get_tuple(item):
    return item['image'], item['label']

def get_imgs(item):
    return item['image']

model = get_open_shelf_resnet()

model.summary()


train_ds, test_ds, val_ds = get_open_shelf_dataset()

filepath =os.path.join('./models', model.name + "_{epoch:02d}.h5")

callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, embeddings_freq=0, update_freq='batch'),
             tf.keras.callbacks.ModelCheckpoint(filepath)
             ]

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=[None, tf.keras.losses.SparseCategoricalCrossentropy()],
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
              )

train_ds = train_ds.map(resize).shuffle(100).batch(20).map(get_tuple)
val_ds = val_ds.map(resize).batch(20).map(get_tuple)
test_ds = test_ds.map(resize).batch(20).map(get_imgs)

# model.fit(train_ds, epochs=60, validation_data=val_ds, callbacks=callbacks)

# results = model.predict(test_ds)

# print(results)

model.save(os.path.join('./models', model.name + '.h5'))
tf.keras.experimental.export_saved_model(model, os.path.join('./models/' + model.name))

tf.keras.utils.plot_model(model, os.path.join('./models', model.name + '.png'), show_shapes=True)
