import tensorflow as tf

from tensorflow.keras.models import Model
from .losses import GANLoss, PatchNCELoss
from .image_pool import ImagePool
from .networks import Generator, Discriminator, Encoder, PatchSampleMLP
from tensorflow.keras import mixed_precision


class DCL_model(Model):
    """ Create a DCLGAN model"""

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
                 fp16=False,
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
                                 norm_layer, use_antialias, resnet_blocks, impl, fp16)
        self.netG_BA = Generator(source_shape, target_shape,
                                 norm_layer, use_antialias, resnet_blocks, impl, fp16)
        self.netD_A = Discriminator(
            target_shape, norm_layer, use_antialias, impl, fp16)
        self.netD_B = Discriminator(
            target_shape, norm_layer, use_antialias, impl, fp16)
        self.netE_A = Encoder(self.netG_AB, self.nce_layers, fp16)
        self.netE_B = Encoder(self.netG_BA, self.nce_layers, fp16)
        self.netF_A = PatchSampleMLP(netF_units, netF_num_patches, fp16)
        self.netF_B = PatchSampleMLP(netF_units, netF_num_patches, fp16)
        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(pool_size)
        # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(pool_size)
        self.fp16 = fp16
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

        def get_wrapped_optimizer(optimizer):
            if self.fp16:
                return mixed_precision.LossScaleOptimizer(optimizer)
            else:
                return optimizer
        self.G_AB_optimizer = get_wrapped_optimizer(G_AB_optimizer)
        self.G_BA_optimizer = get_wrapped_optimizer(G_BA_optimizer)
        self.F_A_optimizer = get_wrapped_optimizer(F_A_optimizer)
        self.F_B_optimizer = get_wrapped_optimizer(F_B_optimizer)
        self.D_A_optimizer = get_wrapped_optimizer(D_A_optimizer)
        self.D_B_optimizer = get_wrapped_optimizer(D_B_optimizer)
        self.gan_loss_func = GANLoss(self.gan_mode)
        self.nce_loss_func = PatchNCELoss(self.nce_temp, self.nce_lambda)
        self.idt_loss_func = tf.keras.losses.MeanAbsoluteError()

    def call(self, inputs, training=None, mask=None):  # for computing output shape
        return self.netG_AB(inputs)

    @tf.function(jit_compile=True)
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
            if self.fp16:
                scaled_D_A_loss = self.D_A_optimizer.get_scaled_loss(D_A_loss)
                scaled_D_B_loss = self.D_B_optimizer.get_scaled_loss(D_B_loss)
                scaled_G_AB_loss = self.G_AB_optimizer.get_scaled_loss(G_loss)
                scaled_G_BA_loss = self.G_BA_optimizer.get_scaled_loss(G_loss)
                scaled_F_A_loss = self.F_A_optimizer.get_scaled_loss(NCE_loss)
                scaled_F_B_loss = self.F_B_optimizer.get_scaled_loss(NCE_loss)

        """ Apply Gradients """
        if self.fp16:
            scaled_D_A_loss_grads = tape.gradient(
                scaled_D_A_loss, self.netD_A.trainable_variables)
            D_A_loss_grads = self.D_A_optimizer.get_unscaled_gradients(
                scaled_D_A_loss_grads)
        else:
            D_A_loss_grads = tape.gradient(
                D_A_loss, self.netD_A.trainable_variables)
        self.D_A_optimizer.apply_gradients(
            zip(D_A_loss_grads, self.netD_A.trainable_variables))

        if self.fp16:
            scaled_D_B_loss_grads = tape.gradient(
                scaled_D_B_loss, self.netD_B.trainable_variables)
            D_B_loss_grads = self.D_B_optimizer.get_unscaled_gradients(
                scaled_D_B_loss_grads)
        else:
            D_B_loss_grads = tape.gradient(
                D_B_loss, self.netD_B.trainable_variables)
        self.D_B_optimizer.apply_gradients(
            zip(D_B_loss_grads, self.netD_B.trainable_variables))

        if self.fp16:
            scaled_G_AB_loss_grads = tape.gradient(
                scaled_G_AB_loss, self.netG_AB.trainable_variables)
            G_AB_loss_grads = self.G_AB_optimizer.get_unscaled_gradients(
                scaled_G_AB_loss_grads)
        else:
            G_AB_loss_grads = tape.gradient(
                G_loss, self.netG_AB.trainable_variables)
        self.G_AB_optimizer.apply_gradients(
            zip(G_AB_loss_grads, self.netG_AB.trainable_variables))

        if self.fp16:
            scaled_G_BA_loss_grads = tape.gradient(
                scaled_G_BA_loss, self.netG_BA.trainable_variables)
            G_BA_loss_grads = self.G_BA_optimizer.get_unscaled_gradients(
                scaled_G_BA_loss_grads)
        else:
            G_BA_loss_grads = tape.gradient(
                G_loss, self.netG_BA.trainable_variables)
        self.G_BA_optimizer.apply_gradients(
            zip(G_BA_loss_grads, self.netG_BA.trainable_variables))

        if self.fp16:
            scaled_F_A_loss_grads = tape.gradient(
                scaled_F_A_loss, self.netF_A.trainable_variables)
            F_A_loss_grads = self.F_A_optimizer.get_unscaled_gradients(
                scaled_F_A_loss_grads)
        else:
            F_A_loss_grads = tape.gradient(
                NCE_loss, self.netF_A.trainable_variables)
        self.F_A_optimizer.apply_gradients(
            zip(F_A_loss_grads, self.netF_A.trainable_variables))

        if self.fp16:
            scaled_F_B_loss_grads = tape.gradient(
                scaled_F_B_loss, self.netF_B.trainable_variables)
            F_B_loss_grads = self.F_B_optimizer.get_unscaled_gradients(
                scaled_F_B_loss_grads)
        else:
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
