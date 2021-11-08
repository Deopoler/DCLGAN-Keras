""" Implement the following components that used in CUT/FastCUT model.
Generator (Resnet-based)
Discriminator (PatchGAN)
Encoder
PatchSampleMLP
CUT_model
"""

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.python.ops.nn_ops import pool

from .layers import ConvBlock, ConvTransposeBlock, ResBlock, AntialiasSampling, Padding2D
from .losses import GANLoss, PatchNCELoss
from .image_pool import ImagePool


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


class DCL_model(Model):
    """ Create a CUT/FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020 (https://arxiv.org/abs/2007.15651).
    """

    def __init__(self,
                 source_shape,
                 target_shape,
                 dclgan_mode='dclgan',
                 gan_mode='lsgan',
                 use_antialias=True,
                 norm_layer='instance',
                 resnet_blocks=9,
                 netF_units=256,
                 netF_num_patches=256,
                 nce_temp=0.07,
                 nce_layers=[3, 5, 7, 11],
                 pool_size=50,
                 impl='ref',
                 **kwargs):
        assert dclgan_mode in ['dclgan']
        assert gan_mode in ['lsgan', 'nonsaturating']
        assert norm_layer in [None, 'batch', 'instance']
        assert netF_units > 0
        assert netF_num_patches > 0
        super(DCL_model, self).__init__(self, **kwargs)

        self.gan_mode = gan_mode
        self.nce_temp = nce_temp
        self.nce_layers = nce_layers
        self.netG_AB = Generator(source_shape, target_shape,
                                 norm_layer, use_antialias, resnet_blocks, impl)
        self.netG_BA = Generator(source_shape, target_shape,
                                 norm_layer, use_antialias, resnet_blocks, impl)
        self.netD_A = Discriminator(
            target_shape, norm_layer, use_antialias, impl)
        self.netD_B = Discriminator(
            target_shape, norm_layer, use_antialias, impl)
        self.netE_A = Encoder(self.netG_AB, self.nce_layers)
        self.netE_B = Encoder(self.netG_BA, self.nce_layers)
        self.netF_A = PatchSampleMLP(netF_units, netF_num_patches)
        self.netF_B = PatchSampleMLP(netF_units, netF_num_patches)
        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(pool_size)
        # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(pool_size)
        self.pool_size = pool_size

        if dclgan_mode == 'dclgan':
            self.nce_lambda = 2.0
            self.use_identity = True
        else:
            raise ValueError(dclgan_mode)

    def compile(self,
                G_AB_optimizer,
                G_BA_optimizer,
                F_A_optimizer,
                F_B_optimizer,
                D_A_optimizer,
                D_B_optimizer,):
        super(DCL_model, self).compile()
        self.G_AB_optimizer = G_AB_optimizer
        self.G_BA_optimizer = G_BA_optimizer
        self.F_A_optimizer = F_A_optimizer
        self.F_B_optimizer = F_B_optimizer
        self.D_A_optimizer = D_A_optimizer
        self.D_B_optimizer = D_B_optimizer
        self.gan_loss_func = GANLoss(self.gan_mode)
        self.nce_loss_func = PatchNCELoss(self.nce_temp, self.nce_lambda)
        self.idt_loss_func = tf.keras.losses.MeanAbsoluteError()

    def call(self, inputs, training=None, mask=None):  # for computing output shape
        return self.netG_AB(inputs)

    @tf.function
    def train_step(self, batch_data):
        # A is source and B is target
        real_A, real_B = batch_data
        real = tf.concat([real_A, real_B], axis=0)

        with tf.GradientTape(persistent=True) as tape:

            fake_AB = self.netG_AB(real, training=True)
            fake_B = fake_AB[:real_A.shape[0]]  # Input: real_A
            if self.use_identity:
                idt_B = fake_AB[real_A.shape[0]:]  # Input: real_B

            fake_BA = self.netG_BA(real, training=True)
            fake_A = fake_BA[real_A.shape[0]:]  # Input: real_B
            if self.use_identity:
                idt_A = fake_BA[:real_A.shape[0]]  # Input: real_A

            """Calculate GAN loss for discriminator D_A"""

            fake_A_ = self.fake_A_pool.query(fake_A)

            fake_A_score = self.netD_A(fake_A_, training=True)
            D_A_fake_loss = tf.reduce_mean(
                self.gan_loss_func(fake_A_score, False))

            real_A_score = self.netD_A(real_A, training=True)
            D_A_real_loss = tf.reduce_mean(
                self.gan_loss_func(real_A_score, True))
            D_A_loss = (D_A_fake_loss + D_A_real_loss) * 0.5

            """Calculate GAN loss for discriminator D_B"""
            fake_B_ = self.fake_B_pool.query(fake_B)
            fake_B_score = self.netD_B(fake_B_, training=True)
            D_B_fake_loss = tf.reduce_mean(
                self.gan_loss_func(fake_B_score, False))

            real_B_score = self.netD_B(real_B, training=True)
            D_B_real_loss = tf.reduce_mean(
                self.gan_loss_func(real_B_score, True))
            D_B_loss = (D_B_fake_loss + D_B_real_loss) * 0.5

            """Calculate GAN loss and NCE loss for the generator"""
            fake_B_score_ = self.netD_B(fake_B, training=True)
            G_AB_GAN_loss = tf.reduce_mean(
                self.gan_loss_func(fake_B_score_, True))

            fake_A_score_ = self.netD_A(fake_A, training=True)
            G_BA_GAN_loss = tf.reduce_mean(
                self.gan_loss_func(fake_A_score_, True))

            G_GAN_loss = (G_AB_GAN_loss + G_BA_GAN_loss) * 0.5

            NCE_loss1 = self.nce_loss_func(
                real_A, fake_B, self.netE_A, self.netE_B, self.netF_A, self.netF_B)
            NCE_loss2 = self.nce_loss_func(
                real_B, fake_A, self.netE_B, self.netE_A, self.netF_B, self.netF_A)
            NCE_loss = (NCE_loss1 + NCE_loss2) * 0.5

            if self.use_identity:
                idt_A_loss = self.idt_loss_func(idt_B, real_B)
                idt_B_loss = self.idt_loss_func(idt_A, real_A)
                NCE_loss += (idt_A_loss + idt_B_loss) * 0.5

            G_loss = G_GAN_loss + NCE_loss

        """ Apply Gradients """
        D_A_loss_grads = tape.gradient(
            D_A_loss, self.netD_A.trainable_variables)
        self.D_A_optimizer.apply_gradients(
            zip(D_A_loss_grads, self.netD_A.trainable_variables))

        D_B_loss_grads = tape.gradient(
            D_B_loss, self.netD_B.trainable_variables)
        self.D_B_optimizer.apply_gradients(
            zip(D_B_loss_grads, self.netD_B.trainable_variables))

        G_AB_loss_grads = tape.gradient(
            G_loss, self.netG_AB.trainable_variables)
        self.G_AB_optimizer.apply_gradients(
            zip(G_AB_loss_grads, self.netG_AB.trainable_variables))

        G_BA_loss_grads = tape.gradient(
            G_loss, self.netG_BA.trainable_variables)
        self.G_BA_optimizer.apply_gradients(
            zip(G_BA_loss_grads, self.netG_BA.trainable_variables))

        F_A_loss_grads = tape.gradient(
            NCE_loss, self.netF_A.trainable_variables)
        self.F_A_optimizer.apply_gradients(
            zip(F_A_loss_grads, self.netF_A.trainable_variables))

        F_B_loss_grads = tape.gradient(
            NCE_loss, self.netF_B.trainable_variables)
        self.F_B_optimizer.apply_gradients(
            zip(F_B_loss_grads, self.netF_B.trainable_variables))

        del tape
        return {'D_A_loss': D_A_loss,
                'D_B_loss': D_B_loss,
                'G_AB_GAN_loss': G_AB_GAN_loss,
                'G_BA_GAN_loss': G_BA_GAN_loss,
                'NCE_loss': NCE_loss}
