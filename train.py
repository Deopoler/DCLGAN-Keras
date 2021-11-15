""" USAGE
python ./train.py --train_a_dir ./datasets/horse2zebra/trainA --train_b_dir ./datasets/horse2zebra/trainB --test_a_dir ./datasets/horse2zebra/testA --test_b_dir ./datasets/horse2zebra/testB
"""

import os
import argparse
import datetime
import tensorflow as tf
from modules.dcl_model import DCL_model
from modules.simdcl_model import SimDCL_model
from utils import GANMonitor, create_dataset
try:
    import wandb
    from wandb.keras import WandbCallback
except ImportError:
    pass


def ArgParse():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CUT training usage.')
    # Training
    parser.add_argument('--mode', help="Model's mode be one of: 'dclgan', 'simdcl'",
                        type=str, default='dclgan', choices=['dclgan', 'simdcl'])
    parser.add_argument(
        '--epochs', help='Number of training epochs', type=int, default=200)
    parser.add_argument(
        '--batch_size', help='Training batch size', type=int, default=1)
    parser.add_argument(
        '--beta_1', help='First Momentum term of adam', type=float, default=0.5)
    parser.add_argument(
        '--beta_2', help='Second Momentum term of adam', type=float, default=0.999)
    parser.add_argument(
        '--lr', help='Initial learning rate for adam', type=float, default=0.0002)
    parser.add_argument('--lr_decay_rate',
                        help='lr_decay_rate', type=float, default=0.9)
    parser.add_argument('--lr_decay_step',
                        help='lr_decay_step', type=int, default=100000)
    # Define data
    parser.add_argument('--out_dir', help='Outputs folder',
                        type=str, default='./output')
    parser.add_argument('--train_a_dir', help='Train-A dataset folder',
                        type=str, default='./datasets/horse2zebra/trainA')
    parser.add_argument('--train_b_dir', help='Train-B dataset folder',
                        type=str, default='./datasets/horse2zebra/trainB')
    parser.add_argument('--test_a_dir', help='Test-A dataset folder',
                        type=str, default='./datasets/horse2zebra/testA')
    parser.add_argument('--test_b_dir', help='Test-B dataset folder',
                        type=str, default='./datasets/horse2zebra/testB')
    # Misc
    parser.add_argument(
        '--ckpt', help='Resume training from checkpoint', type=str)
    parser.add_argument(
        '--save_n_epoch', help='Every n epochs to save checkpoints', type=int, default=5)
    parser.add_argument('--impl', help="(Faster)Custom op use:'cuda'; (Slower)Tensorflow op use:'ref'",
                        type=str, default='ref', choices=['ref', 'cuda'])

    parser.add_argument('--logger', help="Logger be one of: 'tensorboard', 'wandb'",
                        type=str, default='tensorboard', choices=['tensorboard', 'wandb'])

    args = parser.parse_args()

    # Check arguments
    assert args.lr > 0
    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.save_n_epoch > 0
    assert os.path.exists(
        args.train_a_dir), 'Error: Train A dataset does not exist.'
    assert os.path.exists(
        args.train_b_dir), 'Error: Train B dataset does not exist.'
    assert os.path.exists(
        args.test_a_dir), 'Error: Test A dataset does not exist.'
    assert os.path.exists(
        args.test_b_dir), 'Error: Test B dataset does not exist.'

    return args


def main(args):
    # Create datasets
    train_dataset, test_dataset = create_dataset(args.train_a_dir,
                                                 args.train_b_dir,
                                                 args.test_a_dir,
                                                 args.test_b_dir,
                                                 args.batch_size)

    # Get image shape
    source_image, target_image = next(iter(train_dataset))
    source_shape = source_image.shape[1:]
    target_shape = target_image.shape[1:]

    # Create model
    if args.mode == 'dclgan':
        dclgan = DCL_model(source_shape, target_shape,
                           dclgan_mode=args.mode, impl=args.impl)
    else:
        dclgan = SimDCL_model(source_shape, target_shape,
                              simdcl_mode=args.mode, impl=args.impl)
    # without this, error occurs, when saving model
    dclgan.compute_output_shape((None, *target_shape))

    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.lr,
                                                                 decay_steps=args.lr_decay_step,
                                                                 decay_rate=args.lr_decay_rate,
                                                                 staircase=True)

    # Compile model
    dclgan.compile(
        G_AB_optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2),
        G_BA_optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2),
        F_A_optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2),
        F_B_optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2),
        D_A_optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2),
        D_B_optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2),
    )

    # Restored from previous checkpoints, or initialize checkpoints from scratch
    if args.ckpt is not None:
        latest_ckpt = tf.train.latest_checkpoint(args.ckpt)
        dclgan.load_weights(latest_ckpt)
        initial_epoch = int(latest_ckpt[-3:])
        print(f"Restored from {latest_ckpt}.")
    else:
        initial_epoch = 0
        print("Initializing from scratch...")

    # Create folders to store the output information
    result_dir = f'{args.out_dir}/images'
    checkpoint_dir = f'{args.out_dir}/checkpoints'
    log_dir = f'{args.out_dir}/logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    # Create validating callback to generate output image every epoch
    plotter_callback = GANMonitor(
        dclgan.netG_AB, dclgan.netG_BA, test_dataset, result_dir, args.logger)

    # Create checkpoint callback to save model's checkpoints every n epoch (default 5)
    # "period" to save every n epochs, "save_freq" to save every n batches
    dataset_len = tf.data.experimental.cardinality(train_dataset).numpy()
    period = args.save_n_epoch
    save_freq = int(dataset_len * period)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir+'/{epoch:03d}', save_freq=save_freq, verbose=0, save_weights_only=True)

    if args.logger == 'tensorboard':
        # Create tensorboard callback to log losses every epoch
        logger_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    else:
        wandb.init(config=args)
        # Create wandb callback to log losses every epoch
        logger_callback = WandbCallback()

    # Train cut model
    dclgan.fit(train_dataset,
               epochs=args.epochs,
               initial_epoch=initial_epoch,
               callbacks=[plotter_callback,
                          checkpoint_callback, logger_callback],
               verbose=1)


if __name__ == '__main__':
    main(ArgParse())
