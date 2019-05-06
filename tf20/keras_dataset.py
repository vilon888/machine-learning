from tensorflow import config

physical_devices = config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    config.experimental.set_memory_growth(device=gpu, enable=True)

import tensorflow_datasets as tfds
import tensorflow as tf

from datasets.coco import Coco2017


# See available datasets
print(tfds.list_builders())

# Construct a tf.data.Dataset
ds_train, ds_val, ds_test = tfds.load(name="coco2017", split=["train", "validation", "test"])

# Build your input pipeline
# ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
ds_train = ds_train.shuffle(1000).prefetch(10)
for features in ds_train.take(1):
  images, filenames, objects = features["image"], features["image/filename"], features['objects']