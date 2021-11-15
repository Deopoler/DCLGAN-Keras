import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Reshape
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.models import Model

from .layers import ConvBlock, ConvTransposeBlock, ResBlock, AntialiasSampling, Padding2D


def Generator(input_shape, output_shape, norm_layer, use_antialias, resnet_blocks, impl):
    """ Create a Resnet-based generator.
    Adapt from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style).
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics. 
    """
    use_bias = (norm_layer == 'instance')

    inputs = Input(shape=input_shape)
    x = Padding2D(3, pad_type='reflect')(inputs)
    x = ConvBlock(64, 7, padding='valid', use_bias=use_bias,
                  norm_layer=norm_layer, activation='relu')(x)

    if use_antialias:
        x = ConvBlock(128, 3, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation='relu')(x)
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
        x = ConvBlock(256, 3, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation='relu')(x)
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
    else:
        x = ConvBlock(128, 3, strides=2, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation='relu')(x)
        x = ConvBlock(256, 3, strides=2, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation='relu')(x)

    for _ in range(resnet_blocks):
        x = ResBlock(256, 3, use_bias, norm_layer)(x)

    if use_antialias:
        x = AntialiasSampling(4, mode='up', impl=impl)(x)
        x = ConvBlock(128, 3, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation='relu')(x)
        x = AntialiasSampling(4, mode='up', impl=impl)(x)
        x = ConvBlock(64, 3, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation='relu')(x)
    else:
        x = ConvTransposeBlock(128, 3, strides=2, padding='same',
                               use_bias=use_bias, norm_layer=norm_layer, activation='relu')(x)
        x = ConvTransposeBlock(64, 3, strides=2, padding='same',
                               use_bias=use_bias, norm_layer=norm_layer, activation='relu')(x)

    x = Padding2D(3, pad_type='reflect')(x)
    outputs = ConvBlock(output_shape[-1], 7,
                        padding='valid', activation='tanh')(x)

    return Model(inputs=inputs, outputs=outputs, name='generator')


def Discriminator(input_shape, norm_layer, use_antialias, impl):
    """ Create a PatchGAN discriminator.
    PatchGAN classifier described in the original pix2pix paper (https://arxiv.org/abs/1611.07004).
    Such a patch-level discriminator architecture has fewer parameters
    than a full-image discriminator and can work on arbitrarily-sized images
    in a fully convolutional fashion.
    """
    use_bias = (norm_layer == 'instance')

    inputs = Input(shape=input_shape)

    if use_antialias:
        x = ConvBlock(64, 4, padding='same',
                      activation=tf.nn.leaky_relu)(inputs)
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
        x = ConvBlock(128, 4, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
        x = ConvBlock(256, 4, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
    else:
        x = ConvBlock(64, 4, strides=2, padding='same',
                      activation=tf.nn.leaky_relu)(inputs)
        x = ConvBlock(128, 4, strides=2, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)
        x = ConvBlock(256, 4, strides=2, padding='same', use_bias=use_bias,
                      norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)

    x = Padding2D(1, pad_type='constant')(x)
    x = ConvBlock(512, 4, padding='valid', use_bias=use_bias,
                  norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)
    x = Padding2D(1, pad_type='constant')(x)
    outputs = ConvBlock(1, 4, padding='valid')(x)

    return Model(inputs=inputs, outputs=outputs, name='discriminator')


def Encoder(generator, nce_layers):
    """ Create an Encoder that shares weights with the generator.
    """
    assert max(nce_layers) <= len(generator.layers) and min(nce_layers) >= 0

    outputs = [generator.get_layer(index=idx).output for idx in nce_layers]

    return Model(inputs=generator.input, outputs=outputs, name='encoder')


class PatchSampleMLP(Model):
    """ Create a PatchSampleMLP.
    Adapt from official CUT implementation (https://github.com/taesungp/contrastive-unpaired-translation).
    PatchSampler samples patches from pixel/feature-space.
    Two-layer MLP projects both the input and output patches to a shared embedding space.
    """

    def __init__(self, units, num_patches, **kwargs):
        super(PatchSampleMLP, self).__init__(**kwargs)
        self.units = units
        self.num_patches = num_patches
        self.l2_norm = Lambda(
            lambda x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 10-10))

    def build(self, input_shape):
        initializer = tf.random_normal_initializer(0., 0.02)
        feats_shape = input_shape
        for feat_id in range(len(feats_shape)):
            mlp = tf.keras.models.Sequential([
                Dense(self.units, activation="relu",
                      kernel_initializer=initializer),
                Dense(self.units, kernel_initializer=initializer),
            ])
            setattr(self, f'mlp_{feat_id}', mlp)

    def call(self, inputs, patch_ids=None, training=None):
        feats = inputs
        samples = []
        ids = []
        for feat_id, feat in enumerate(feats):
            B, H, W, C = feat.shape

            feat_reshape = tf.reshape(feat, [B, -1, C])

            if patch_ids is not None:
                patch_id = patch_ids[feat_id]
            else:
                patch_id = tf.random.shuffle(
                    tf.range(H * W))[:min(self.num_patches, H * W)]

            x_sample = tf.reshape(
                tf.gather(feat_reshape, patch_id, axis=1), [-1, C])
            mlp = getattr(self, f'mlp_{feat_id}')
            x_sample = mlp(x_sample)
            x_sample = self.l2_norm(x_sample)
            samples.append(x_sample)
            ids.append(patch_id)

        return samples, ids


def MappingMLP(in_layer, num_patches, nc, dim):
    """ Create a MappingMLP.
    Adapt from official DCLGAN implementation (https://github.com/JunlinHan/DCLGAN).
    """
    inputs = Input(shape=(in_layer, num_patches, nc))
    x = Reshape((1, in_layer, num_patches, nc))(inputs)
    x = Conv2D(dim, 3, strides=2, activation='relu')(x)
    x = AdaptiveAveragePooling2D(1)(x)
    x = Flatten()(x)
    x = Dense(dim, activation='relu')(x)
    outputs = Dense(dim)(x)

    return Model(inputs=inputs, outputs=outputs, name='mapping_mlp')
