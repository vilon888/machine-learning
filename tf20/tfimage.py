import tensorflow as tf
import cv2

img = cv2.imread('/home/vilon_tao/Projects/machine-learning/tf20/images/0-augmentation.jpg')

im = tf.image.random_crop(img, [20,20,3])

cv2.imshow('tfimg', im.numpy())
pass