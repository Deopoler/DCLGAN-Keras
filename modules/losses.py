""" Implement the following loss functions that used in CUT/FastCUT model.
GANLoss
PatchNCELoss
"""

import tensorflow as tf


class GANLoss:
    def __init__(self, gan_mode):
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = tf.keras.losses.MeanSquaredError()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented.')

    def __call__(self, prediction, target_is_real):

        if self.gan_mode == 'lsgan':
            if target_is_real:
                loss = self.loss(tf.ones_like(prediction), prediction)
            else:
                loss = self.loss(tf.zeros_like(prediction), prediction)

        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = tf.reduce_mean(tf.math.softplus(-prediction))
            else:
                loss = tf.reduce_mean(tf.math.softplus(prediction))

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = tf.reduce_mean(-prediction)
            else:
                loss = tf.reduce_mean(prediction)
        return loss


class PatchNCELoss:
    def __init__(self, nce_temp, nce_lambda):
        # Potential: only supports for batch_size=1 now.
        self.nce_temp = nce_temp
        self.nce_lambda = nce_lambda
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            from_logits=True)
        self.cos = tf.keras.losses.CosineSimilarity(
            axis=-1,
            reduction=tf.keras.losses.Reduction.NONE,
        )

    def __call__(self, source, target, netE_source, netE_target, netF_source, netF_target):
        feat_source = netE_source(source, training=True)
        feat_target = netE_target(target, training=True)

        feat_source_pool, sample_ids = netF_source(
            feat_source, patch_ids=None, training=True)
        feat_target_pool, _ = netF_target(
            feat_target, patch_ids=sample_ids, training=True)

        total_nce_loss = 0.0
        for feat_s, feat_t in zip(feat_source_pool, feat_target_pool):
            n_patches, dim = feat_s.shape

            logit = -self.cos(tf.reshape(feat_s, shape=(n_patches, 1, dim)),
                              tf.reshape(feat_t, shape=(1, n_patches, dim))) / self.nce_temp
    # Diagonal entries are pos logits, the others are neg logits.
            diagonal = tf.eye(n_patches, dtype=tf.bool)
            target = tf.where(diagonal, 1.0, 0.0)

            loss = self.cross_entropy_loss(target, logit) * self.nce_lambda
            total_nce_loss += tf.reduce_mean(loss)

        return total_nce_loss / len(feat_source_pool)
