import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
# import tensorflow.compat.v2 as tf
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from MyDataset import MyDataset

from tensorflow import config

physical_devices = config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    config.experimental.set_memory_growth(device=gpu, enable=True)


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
        builder_kwargs=dict(dataset_name='open.shelf.classfication'),
        download_and_prepare_kwargs=dict(download_config=dl_config),
    )

    return train_ds, test_ds, val_ds


def am_softmax_loss(label, prediction, m=0.35, s=30):

    cos_theta = tf.clip_by_value(prediction,-1,1)
    phi = cos_theta - m

    label1 = tf.cast(tf.keras.backend.flatten(label), 'int64')

    label_onehot = tf.one_hot(label1,7)
    adjust_theta = s * tf.where(tf.equal(label_onehot,1),phi,cos_theta)
    loss = tf.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    return loss(label_onehot, adjust_theta)


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
                                      # outputs=[embedding, prediction],
                                      outputs=prediction,
                                      name='open_shelf_resnet_model')

    return open_shelf_model


def resize(item):
    H = tf.cast(item['height'], dtype=tf.float32)
    W = tf.cast(item['width'], dtype=tf.float32)
    scale = tf.random.uniform([], minval=0.8, maxval=1, dtype=tf.float32)

    new_H = tf.cast(H * scale, dtype=tf.int32)
    new_W = tf.cast(W * scale, dtype=tf.int32)
    item['image'] = tf.image.random_crop(item['image'], size=(new_H, new_W, 3))

    item['image'] = tf.image.resize(item['image'], (96, 96))
    item['image'] = tf.cast(item['image'] / 255, dtype=tf.float32)
    return item


def resize_val_test(item):
    item['image'] = tf.image.resize(item['image'], (96, 96))
    item['image'] = tf.cast(item['image'] / 255, dtype=tf.float32)
    return item


def get_tuple(item):
    return item['image'], item['label'] #(item['label'], item['label'])


def get_imgs(item):
    return item['image']

NUM_TRAIN_IMAGES = 60000

def create_model():
  max_pool = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same")
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential([
      tf.keras.layers.Reshape(
          target_shape=[28, 28, 1],
          input_shape=(28, 28,)),
      tf.keras.layers.Conv2D(2, 5, padding="same", activation=tf.nn.relu),
      max_pool,
      tf.keras.layers.Conv2D(4, 5, padding="same", activation=tf.nn.relu),
      max_pool,
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(10)])


def compute_loss(logits, labels):
  loss = tf.reduce_sum(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))
  # Scale loss by global batch size.
  return loss * (1. / 48)


def mnist_datasets():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  # Numpy defaults to dtype=float64; TF defaults to float32. Stick with float32.
  x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
  y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
  # TODO(priyag): `strategy.make_numpy_iterator` can be used directly instead of
  # converting to datasets.
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  return train_dataset, test_dataset


def train():
  """Run a CNN model on MNIST data to demonstrate DistributedStrategies."""

  # tf.enable_v2_behavior()
  #
  # num_gpus = 3
  # if num_gpus is None:
  #   devices = None
  # elif num_gpus == 0:
  #   devices = ["/device:CPU:0"]
  # else:
  #   devices = ["/device:GPU:{}".format(i) for i in range(num_gpus)]
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():
    # train_ds, test_ds = mnist_datasets()
    train_ds, test_ds, val_ds = get_open_shelf_dataset()
    # train_ds = train_ds.shuffle(NUM_TRAIN_IMAGES).batch(48, drop_remainder=True)
    # test_ds = test_ds.batch(48, drop_remainder=True)

    train_ds = train_ds.map(resize).shuffle(1000).batch(48).map(get_tuple)
    # val_ds = val_ds.map(resize_val_test).batch(48, drop_remainder=True).map(get_tuple)
    # test_ds_eval = test_ds.map(resize_val_test).batch(48, drop_remainder=True).map(get_tuple)
    test_ds = test_ds.map(resize_val_test).batch(48).map(get_imgs)
    #
    # test_ds_model = test_ds.map(resize_val_test).batch(1, drop_remainder=True).map(get_tuple)

    # model = create_model()
    model = get_open_shelf_resnet()
    # optimizer = tf.keras.optimizers.SGD(0.01, 0.5)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4)
    training_loss = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        "training_accuracy", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        "test_accuracy", dtype=tf.float32)

    # @tf.function
    def train_step(inputs):
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        # loss = compute_loss(logits, labels)
        loss = am_softmax_loss(labels, logits)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      training_loss.update_state(loss)
      training_accuracy.update_state(labels, logits)

    # @tf.function
    def test_step(inputs):
      images, labels = inputs
      logits = model(images, training=False)
      # loss = compute_loss(logits, labels)
      loss = am_softmax_loss(labels, logits)
      test_loss.update_state(loss)
      test_accuracy.update_state(labels, logits)

    train_iterator = strategy.make_dataset_iterator(train_ds)
    test_iterator = strategy.make_dataset_iterator(test_ds)

    for epoch in range(0, 5):
      # TODO(b/123315763): Create the tf.function outside this loop once we are
      # able to initialize iterator in eager mode.
      train_ds = strategy.experimental_distribute_dataset(train_ds)
      # dist_train = lambda it: strategy.experimental_run(train_step, it)
      # dist_test = lambda it: strategy.experimental_run(test_step, it)
      # if FLAGS.use_function:
      # dist_train = tf.function(dist_train)
      # dist_test = tf.function(dist_test)

      # Train
      print("Starting epoch {}".format(epoch))

      for item in train_ds:
          strategy.experimental_run_v2(train_step, args=(item,))
      #
      # train_iterator.initialize()
      # while True:
      #   try:
      #     dist_train(train_iterator)
      #   except tf.errors.OutOfRangeError:
      #     print('end one epoch')
      #     break
      print("Training loss: {:0.4f}, accuracy: {:0.2f}%".format(
      training_loss.result(), training_accuracy.result() * 100))
      training_loss.reset_states()
      training_accuracy.reset_states()

      # Test
      test_iterator.initialize()
      while True:
        try:
          dist_test(test_iterator)
        except tf.errors.OutOfRangeError:
          break
      print("Test loss: {:0.4f}, accuracy: {:0.2f}%".format(
          test_loss.result(), test_accuracy.result() * 100))
      test_loss.reset_states()
      test_accuracy.reset_states()


if __name__ == "__main__":
  train()
