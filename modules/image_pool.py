import tensorflow as tf
import random

from tensorflow.python.ops.gen_array_ops import size


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = tf.TensorArray(tf.float32, size=self.pool_size)

    def query(self, image):
        """Return an image from the pool.
        Parameters:
            image: the latest generated image from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return image
        if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
            self.images = self.images.write(self.num_imgs, image)
            self.num_imgs = self.num_imgs + 1
            return image
        else:
            p = random.uniform(0, 1)
            if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                random_id = random.randint(
                    0, self.pool_size - 1)  # randint is inclusive
                tmp = self.images.read(random_id)
                self.images = self.images.write(random_id, image)
                return tmp
            else:       # by another 50% chance, the buffer will return the current image
                return image
