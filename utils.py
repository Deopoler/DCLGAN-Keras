import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
try:
    import wandb
except ImportError:
    pass


def create_dir(dir):
    """ Create the directory.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} createrd')
    else:
        print(f'Directory {dir} already exists')

    return dir


@tf.function
def load_image(image_file, data_augmentation=True):
    """ Load the image file.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    if data_augmentation:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.resize(image, size=(286, 286))
    image = tf.image.random_crop(image, size=(256, 256, 3))
    if tf.shape(image)[-1] == 1:
        image = tf.tile(image, [1, 1, 3])

    return image


def create_dataset(train_a_folder,
                   train_b_folder,
                   test_a_folder,
                   test_b_folder,
                   batch_size):
    """ Create tf.data.Dataset.
    """
    # Create train dataset
    train_src_dataset = tf.data.Dataset.list_files(
        [train_a_folder+'/*.jpg', train_a_folder+'/*.png'], shuffle=True)
    train_src_dataset = (
        train_src_dataset.map(
            load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    train_tar_dataset = tf.data.Dataset.list_files(
        [train_b_folder+'/*.jpg', train_b_folder+'/*.png'], shuffle=True)
    train_tar_dataset = (
        train_tar_dataset.map(
            load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    train_dataset = tf.data.Dataset.zip((train_src_dataset, train_tar_dataset))

    # Create test dataset
    test_src_dataset = tf.data.Dataset.list_files(
        [test_a_folder+'/*.jpg', test_a_folder+'/*.png'])
    test_src_dataset = (
        test_src_dataset.map(
            load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_tar_dataset = tf.data.Dataset.list_files(
        [test_b_folder+'/*.jpg', test_b_folder+'/*.png'])
    test_tar_dataset = (
        test_tar_dataset.map(
            load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_dataset = tf.data.Dataset.zip((test_src_dataset, test_tar_dataset))

    return train_dataset, test_dataset


class GANMonitor(tf.keras.callbacks.Callback):
    """ A callback to generate and save images after each epoch
    """

    def __init__(self, generator_ab, generator_ba, test_dataset, out_dir, logger, num_img=2):
        self.num_img = num_img
        self.generator_ab = generator_ab
        self.generator_ba = generator_ba
        self.test_dataset = test_dataset
        self.logger = logger
        self.out_dir = create_dir(out_dir)

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(self.num_img, 4, figsize=(20, 10))
        [ax[0, i].set_title(title) for i, title in enumerate(
            ['ImageA', "TranslatedB", "ImageB", "TranslatedA"])]
        for i, (imageA, imageB) in enumerate(self.test_dataset.take(self.num_img)):
            translated_b = self.generator_ab(imageA)[0].numpy()
            translated_b = (translated_b * 127.5 + 127.5).astype(np.uint8)
            imageA = (imageA[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            translated_a = self.generator_ba(imageB)[0].numpy()
            translated_a = (translated_a * 127.5 + 127.5).astype(np.uint8)
            imageB = (imageB[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            [ax[i, j].imshow(img) for j, img in enumerate(
                [imageA, translated_b, imageB, translated_a])]
            [ax[i, j].axis("off") for j in range(4)]

        plt.savefig(f'{self.out_dir}/epoch={epoch + 1}.png')

        if self.logger == 'wandb':
            wandb.log({'generated_images': plt})
        plt.close()
