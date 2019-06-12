import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import cv2
from MyDataset import MyDataset
from imgaug import augmenters as iaa
import matplotlib

def data_aug_v2(image, label, prob=0.5):
    image = image.numpy()
    aug = iaa.Sequential([
        iaa.Sometimes(prob, iaa.OneOf([iaa.Multiply((0.4, 1.5)),
                                         iaa.GammaContrast((0.3, 2.0))])),
        iaa.Sometimes(prob, iaa.OneOf([iaa.Grayscale((0, 0.3)),
                                         iaa.ContrastNormalization((0.3, 2.5)),
                                         iaa.AddToHueAndSaturation((-50, 50))])),
        iaa.Sometimes(prob, iaa.OneOf([iaa.AverageBlur(k=((1, 4), (1, 4))),
                                         iaa.GaussianBlur(sigma=(0.0, 1.5)),
                                         iaa.MedianBlur(k=(1, 5)),
                                         iaa.MotionBlur(k=[5, 19], angle=(0, 360))])),
        iaa.Sometimes(prob, iaa.AdditiveGaussianNoise(scale=(0, 10), per_channel=0.5))
    ])

    image = aug.augment_image(image)
    image = (image / 255).astype(dtype=np.float32)
    return image, label


def get_open_shelf_dataset():
    dl_config = tfds.download.DownloadConfig(
        manual_dir='/Data/tensorflow_datasets/downloads/manual',
        download_mode=tfds.GenerateMode.REUSE_DATASET_IF_EXISTS
    )

    train_ds, test_ds, val_ds = tfds.load(
        name='my_dataset',
        data_dir='/Data/tensorflow_datasets',
        split=["train", "test", "val"],
        download=False,
        # as_supervised=True,
        builder_kwargs=dict(dataset_name='open.shelf.classfication'),
        download_and_prepare_kwargs=dict(download_config=dl_config),
    )

    return train_ds, test_ds, val_ds

train_ds, test_ds, val_ds = get_open_shelf_dataset()


def resize_eager_test(image, label):
    image = cv2.resize(image, (96, 96))
    image = (image / 255).astype(dtype=np.float32)
    return image, label


def resize(image, label):
    image = cv2.resize(image.numpy(), (96, 96))
    return image, label

# val_ds = val_ds.map(lambda item: tf.numpy_function(
#   resize_eager_test, [item['image'], item['label']], [tf.float32, tf.int64])
# )


val_ds = val_ds \
    .map(lambda item: tf.py_function(resize, [item['image'], item['label']], [tf.uint8, tf.int64])) \
    .map(lambda image, label: tf.py_function(data_aug_v2, [image, label], [tf.float32, tf.int64])) \
    .batch(batch_size=1)


index = 0
for i in val_ds:
    matplotlib.image.imsave('./images/{}-augmentation.jpg'.format(index), i[0][0].numpy())
    index += 1



