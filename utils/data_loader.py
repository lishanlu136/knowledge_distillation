#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/1/3 13:51
@Author     : Li Shanlu
@File       : data_loader.py
@Software   : PyCharm
@Description: A TensorFlow Dataset API based loader for classification problems.
"""
import tensorflow as tf
import random
import cv2


class DataLoader(object):

    def __init__(self, image_paths_list, label_list, image_size, num_classes=2, crop_percent=0.8, channels=3, seed=None):
        """
        Initializes the data loader object
        Args:
            image_paths_list: List of paths of train images.
            label_list: List of paths of train masks (segmentation masks)
            image_size: Tuple of (Height, Width), the final height to resize images to.
            crop_percent: Float in the range 0-1, defining percentage of image to randomly crop.
            channels: int, the number of channels in images.
            seed: An int, if not specified, chosen randomly. Used as the seed for the RNG in the
                  data pipeline.
        """
        self.image_paths_list = image_paths_list
        self.label_list = label_list
        self.image_size = image_size
        self.crop_percent = tf.constant(crop_percent, tf.float32)
        self.channels = channels
        self.num_classes = num_classes
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed

    def _corrupt_brightness(self, image, mask):
        """
        Radnomly applies a random brightness change.
        """
        cond_brightness = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
            image, 0.1), lambda: tf.identity(image))
        return image, mask

    def _corrupt_contrast(self, image, mask):
        """
        Randomly applies a random contrast change.
        """
        cond_contrast = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
            image, 0.2, 1.8), lambda: tf.identity(image))
        return image, mask

    def _corrupt_saturation(self, image, mask):
        """
        Randomly applies a random saturation change.
        """
        cond_saturation = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
            image, 0.2, 1.8), lambda: tf.identity(image))
        return image, mask

    def _crop_random(self, image, mask):
        """
        Randomly crops image and mask in accord.
        """
        cond_crop_image = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32, seed=self.seed), tf.bool)
        cond_crop_mask = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32, seed=self.seed), tf.bool)

        shape = tf.cast(tf.shape(image), tf.float32)
        h = tf.cast(shape[0] * self.crop_percent, tf.int32)
        w = tf.cast(shape[1] * self.crop_percent, tf.int32)

        image = tf.cond(cond_crop_image, lambda: tf.random_crop(
            image, [h, w, self.channels[0]], seed=self.seed), lambda: tf.identity(image))
        mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
            mask, [h, w, self.channels[1]], seed=self.seed), lambda: tf.identity(mask))

        return image, mask

    def _flip_left_right(self, image, mask):
        """
        Randomly flips image and mask left or right in accord.
        """
        image = tf.image.random_flip_left_right(image, seed=self.seed)
        mask = tf.image.random_flip_left_right(mask, seed=self.seed)

        return image, mask

    def _resize_data(self, image, mask):
        """
        Resizes images to specified size.
        """
        image = tf.expand_dims(image, axis=0)
        mask = tf.expand_dims(mask, axis=0)

        image = tf.image.resize_images(image, self.image_size)
        mask = tf.image.resize_nearest_neighbor(mask, self.image_size)

        image = tf.squeeze(image, axis=0)
        mask = tf.squeeze(mask, axis=0)

        return image, mask

    def _random_crop(self, image, label):
        image = tf.random_crop(image, [self.image_size[0], self.image_size[1], 3])
        return image, label

    def _random_flip_left_right(self, image, label):
        image = tf.image.random_flip_left_right(image)
        return image, label

    def _image_standardization(self, image, label):
        image = tf.image.per_image_standardization(image)
        return image, label

    def _pre_resize_image(self, image, label):
        image = tf.py_func(cv2_resize_img, [image], tf.uint8)
        return image, label

    def _resize_image(self, image, label):
        #image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_images(image, self.image_size)
        #image = tf.squeeze(image, axis=0)
        return image, label

    def _parse_data(self, image_paths, labels):
        """
        Reads image and label depending on
        specified exxtension.
        """
        if self.image_size == [160, 160]:
            image_content = tf.read_file(image_paths)
            images = tf.image.decode_image(image_content, channels=self.channels)
        elif self.image_size == [112, 112]:
            images = tf.py_func(read_image_from_cv2, [image_paths], tf.uint8)
        else:
            print("image size error.")
            return None
        return images, labels

    def data_batch(self, augment=True, shuffle=True, batch_size=4, repeat_times=1, num_threads=1, buffer=32):
        """
        Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        Inputs:
            augment: Boolean, whether to augment data or not.
            shuffle: Boolean, whether to shuffle data in buffer or not.
            batch_size: Number of images/masks in each batch returned.
            repeat_times: Number of times to repeat.(train: epoches, val: 1)
            num_threads: Number of parallel subprocesses to load data.
            buffer: Number of images to prefetch in buffer.
        Returns:
            next_element: A tensor with shape [2], where next_element[0]
                          is image batch, next_element[1] is the corresponding
                          mask batch.
            init_op: Data initializer op, needs to be executed in a session
                     for the data queue to be filled up and the next_element op
                     to yield batches.
        """

        # Convert lists of paths to tensors for tensorflow
        images_name_tensor = tf.constant(self.image_paths_list)
        label_tensor = tf.constant(self.label_list)
        # one-hot code
        #label_tensor = tf.one_hot(self.label_list, self.num_classes, axis=-1)
        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices((images_name_tensor, label_tensor))

        # Parse images and labels
        data = data.map(self._parse_data, num_parallel_calls=num_threads).prefetch(150000)

        if self.image_size == [112, 112]:
            data = data.map(self._pre_resize_image, num_parallel_calls=num_threads).prefetch(150000)

        # If augmentation is to be applied
        if augment:
            """
            data = data.map(self._corrupt_brightness,
                            num_parallel_calls=num_threads).prefetch(buffer)

            data = data.map(self._corrupt_contrast,
                            num_parallel_calls=num_threads).prefetch(buffer)

            data = data.map(self._corrupt_saturation,
                            num_parallel_calls=num_threads).prefetch(buffer)

            data = data.map(self._crop_random, num_parallel_calls=num_threads).prefetch(buffer)

            data = data.map(self._flip_left_right,
                            num_parallel_calls=num_threads).prefetch(buffer)
            """

            data = data.map(self._random_crop,
                            num_parallel_calls=num_threads).prefetch(150000)
            data = data.map(self._random_flip_left_right,
                            num_parallel_calls=num_threads).prefetch(150000)

        # Resize to smaller dims for speed
        data = data.map(self._resize_image, num_parallel_calls=num_threads).prefetch(150000)
        data = data.map(self._image_standardization, num_parallel_calls=num_threads).prefetch(150000)

        if shuffle:
            data = data.shuffle(buffer)
        # repeat
        data = data.repeat(repeat_times)
        # Batch the data
        data = data.batch(batch_size)

        # Create iterator
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)

        # Next element Op
        next_element = iterator.get_next()

        # Data set init. op
        init_op = iterator.make_initializer(data)

        return next_element, init_op


# for 112x112
def read_image_from_cv2(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def cv2_resize_img(image):
    image = cv2.resize(image, (128, 128))
    return image
