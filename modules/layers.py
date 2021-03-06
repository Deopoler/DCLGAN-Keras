""" Implement the following layers that used in CUT/FastCUT model.
Padding2D
InstanceNorm
AntialiasSampling
ConvBlock
ConvTransposeBlock
ResBlock
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, BatchNormalization, Activation
from .ops.upfirdn_2d import upsample_2d, downsample_2d
from tensorflow_addons.layers import InstanceNormalization


class Padding2D(Layer):
    """ 2D padding layer.
    """

    def __init__(self, padding=(1, 1), pad_type='constant', **kwargs):
        assert pad_type in ['constant', 'reflect', 'symmetric']
        super(Padding2D, self).__init__(**kwargs)
        self.padding = (padding, padding) if type(
            padding) is int else tuple(padding)
        self.pad_type = pad_type

    def call(self, inputs, training=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]

        return tf.pad(inputs, padding_tensor, mode=self.pad_type)

    def compute_output_shape(self, input_shape):
        B, H, W, C = input_shape
        padding_width, padding_height = self.padding
        return (B, H + padding_height * 2, W + padding_width, C)

    def cast_inputs(self, inputs):
       # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)


class AntialiasSampling(tf.keras.layers.Layer):
    """ Down/Up sampling layer with blur-kernel.
    """

    def __init__(self,
                 kernel_size,
                 mode,
                 impl,
                 **kwargs):
        super(AntialiasSampling, self).__init__(**kwargs)
        if(kernel_size == 1):
            self.k = np.array([1., ])
        elif(kernel_size == 2):
            self.k = np.array([1., 1.])
        elif(kernel_size == 3):
            self.k = np.array([1., 2., 1.])
        elif(kernel_size == 4):
            self.k = np.array([1., 3., 3., 1.])
        elif(kernel_size == 5):
            self.k = np.array([1., 4., 6., 4., 1.])
        elif(kernel_size == 6):
            self.k = np.array([1., 5., 10., 10., 5., 1.])
        elif(kernel_size == 7):
            self.k = np.array([1., 6., 15., 20., 15., 6., 1.])
        self.mode = mode
        self.impl = impl

    def call(self, inputs, training=None):
        if self.mode == 'up':
            x = upsample_2d(inputs, k=self.k,
                            data_format='NHWC', impl=self.impl)
        elif self.mode == 'down':
            x = downsample_2d(inputs, k=self.k,
                              data_format='NHWC', impl=self.impl)
        else:
            raise ValueError(f'Unsupported sampling mode: {self.mode}')
        return x

    def compute_output_shape(self, input_shape):
        N, H, W, C = input_shape
        factor = 2
        return [N, H * factor, W * factor, C]

    def cast_inputs(self, inputs):
       # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)


class ConvBlock(Layer):
    """ ConBlock layer consists of Conv2D + Normalization + Activation.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 use_bias=True,
                 norm_layer=None,
                 activation='linear',
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2d = Conv2D(filters,
                             kernel_size,
                             strides,
                             padding,
                             use_bias=use_bias,
                             kernel_initializer=initializer)
        self.activation = Activation(activation)
        if norm_layer == 'batch':
            self.normalization = BatchNormalization()
        elif norm_layer == 'instance':
            self.normalization = InstanceNormalization()
        else:
            self.normalization = tf.identity

    def call(self, inputs, training=None):
        x = self.conv2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)

        return x

    def cast_inputs(self, inputs):
       # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)


class ConvTransposeBlock(Layer):
    """ ConvTransposeBlock layer consists of Conv2DTranspose + Normalization + Activation.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 use_bias=True,
                 norm_layer=None,
                 activation='linear',
                 **kwargs):
        super(ConvTransposeBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.convT2d = Conv2DTranspose(filters,
                                       kernel_size,
                                       strides,
                                       padding,
                                       use_bias=use_bias,
                                       kernel_initializer=initializer)
        self.activation = Activation(activation)
        if norm_layer == 'batch':
            self.normalization = BatchNormalization()
        elif norm_layer == 'instance':
            self.normalization = InstanceNormalization()
        else:
            self.normalization = tf.identity

    def call(self, inputs, training=None):
        x = self.convT2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)

        return x

    def cast_inputs(self, inputs):
       # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)


class ResBlock(Layer):
    """ ResBlock is a ConvBlock with skip connections.
    Original Resnet paper (https://arxiv.org/pdf/1512.03385.pdf).
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 use_bias,
                 norm_layer,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        self.reflect_pad1 = Padding2D(1, pad_type='reflect')
        self.conv_block1 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer,
                                     activation='relu')

        self.reflect_pad2 = Padding2D(1, pad_type='reflect')
        self.conv_block2 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer)

    def call(self, inputs, training=None):
        x = self.reflect_pad1(inputs)
        x = self.conv_block1(x)

        x = self.reflect_pad2(x)
        x = self.conv_block2(x)

        return inputs + x

    def cast_inputs(self, inputs):
        # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)
