""" Implement the following loss functions that used in CUT/FastCUT model.
GANLoss
PatchNCELoss
PatchNCEAndSimLoss
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


class PatchNCEAndSimLoss:
    """Adapt from official SimDCL implementation (https://github.com/JunlinHan/DCLGAN/blob/main/models/simdcl_model.py).
    """

    def __init__(self, nce_temp, nce_lambda, sim_lambda):
        # Potential: only supports for batch_size=1 now.
        self.nce_temp = nce_temp
        self.nce_lambda = nce_lambda
        self.sim_lambda = sim_lambda
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            from_logits=True)
        self.cos = tf.keras.losses.CosineSimilarity(
            axis=-1,
            reduction=tf.keras.losses.Reduction.NONE,
        )
        self.l1_loss = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM
        )

    def __call__(self, source1, target1, source2, target2, netE_A, netE_B, netF_A, netF_B, netF_1, netF_2, netF_3, netF_4):
        feat_target1 = netE_B(target1, training=True)
        feat_source1 = netE_A(source1, training=True)
        feat_target2 = netE_A(target2, training=True)
        feat_source2 = netE_B(source2, training=True)

        feat_source1_pool, sample_ids = netF_A(
            feat_source1, patch_ids=None, training=True)
        feat_target1_pool, _ = netF_B(
            feat_target1, patch_ids=sample_ids, training=True)
        feat_target1_pool_noid, _ = netF_B(
            feat_target1, patch_ids=None, training=True)

        feat_source2_pool, sample_ids = netF_B(
            feat_source2, patch_ids=None, training=True)
        feat_target2_pool, _ = netF_A(
            feat_target2, patch_ids=sample_ids, training=True)
        feat_target2_pool_noid, _ = netF_A(
            feat_target2, patch_ids=None, training=True)

        total_nce_loss1 = 0.0
        for feat_s, feat_t in zip(feat_source1_pool, feat_target1_pool):
            n_patches, dim = feat_s.shape

            logit = -self.cos(tf.reshape(feat_s, shape=(n_patches, 1, dim)),
                              tf.reshape(feat_t, shape=(1, n_patches, dim))) / self.nce_temp
            # Diagonal entries are pos logits, the others are neg logits.
            diagonal = tf.eye(n_patches, dtype=tf.bool)
            target = tf.where(diagonal, 1.0, 0.0)

            loss = self.cross_entropy_loss(target, logit) * self.nce_lambda
            total_nce_loss1 += tf.reduce_mean(loss)

        total_nce_loss2 = 0.0
        for feat_s, feat_t in zip(feat_source2_pool, feat_target2_pool):
            n_patches, dim = feat_s.shape

            logit = -self.cos(tf.reshape(feat_s, shape=(n_patches, 1, dim)),
                              tf.reshape(feat_t, shape=(1, n_patches, dim))) / self.nce_temp
            # Diagonal entries are pos logits, the others are neg logits.
            diagonal = tf.eye(n_patches, dtype=tf.bool)
            target = tf.where(diagonal, 1.0, 0.0)

            loss = self.cross_entropy_loss(target, logit) * self.nce_lambda
            total_nce_loss2 += tf.reduce_mean(loss)

        total_nce_loss1 = total_nce_loss1 / len(feat_source1_pool)
        total_nce_loss2 = total_nce_loss2 / len(feat_source2_pool)

        nce_loss = (total_nce_loss1 + total_nce_loss2) / 2

        feature_realA = tf.stack(feat_source1_pool, axis=0)
        feature_fakeB = tf.stack(feat_target1_pool_noid, axis=0)
        feature_realB = tf.stack(feat_source2_pool, axis=0)
        feature_fakeA = tf.stack(feat_target2_pool_noid, axis=0)

        feature_realA_out = netF_1(tf.expand_dims(feature_realA, axis=0))
        feature_fakeB_out = netF_2(tf.expand_dims(feature_fakeB, axis=0))
        feature_realB_out = netF_3(tf.expand_dims(feature_realB, axis=0))
        feature_fakeA_out = netF_4(tf.expand_dims(feature_fakeA, axis=0))

        sim_loss = (self.l1_loss(feature_realA_out, feature_fakeA_out) +
                    self.l1_loss(feature_fakeB_out, feature_realB_out)) * self.sim_lambda

        return sim_loss, nce_loss
