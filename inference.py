""" USAGE
python ./inference.py --weights ./output/checkpoints --inputA ./datasets/horse2zebra/testA
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from modules.dcl_model import DCL_model
from utils import create_dir, load_image


def ArgParse():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CUT inference usage.')
    # Inference
    parser.add_argument('--mode', help="Model's mode be one of: 'dclgan', 'simdcl'",
                        type=str, default='dclgan', choices=['dclgan', 'simdcl'])
    parser.add_argument('--weights', help='Pre-trained checkpoints/weights directory',
                        type=str, default='./output/checkpoints')
    parser.add_argument('--inputA', help='Input-A folder',
                        type=str, default='./source/A')
    parser.add_argument('--inputB', help='Input-B folder',
                        type=str, default='./source/B')
    parser.add_argument('--outputA', help='Output-A folder',
                        type=str, default='./translated/A')
    parser.add_argument('--outputB', help='Output-B folder',
                        type=str, default='./translated/B')
    parser.add_argument('--direction', help='')

    args = parser.parse_args()

    # Check arguments
    assert os.path.exists(args.inputA), 'Input-A folder does not exist.'
    assert os.path.exists(args.inputB), 'Input-B folder does not exist.'
    assert os.path.exists(
        args.weights), 'Pre-trained checkpoints/weights does not exist.'
    assert args.output_channel > 0, 'Number of channels must greater than zero.'

    return args


def main(args):
    # Load input images
    input_A_images = tf.data.Dataset.list_files(
        [args.inputA+'/*.jpg', args.inputA+'/*.png'])
    input_A_images = (
        input_A_images.map(lambda x: load_image(x, image_size=None, data_augmentation=False),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(1)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    input_B_images = tf.data.Dataset.list_files(
        [args.inputB+'/*.jpg', args.inputB+'/*.png'])
    input_B_images = (
        input_B_images.map(lambda x: load_image(x, image_size=None, data_augmentation=False),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(1)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    input_images = tf.data.Dataset.zip((input_A_images, input_B_images))
    # Get image shape
    source_image, target_image = next(iter(input_images))
    source_shape = source_image.shape[1:]
    target_shape = target_image.shape[1:]

    # Create model
    dclgan = DCL_model(source_shape, target_shape,
                       dclgan_mode=args.mode)

    # Load weights
    latest_ckpt = tf.train.latest_checkpoint(args.weights)
    dclgan.load_weights(latest_ckpt).expect_partial()
    dclgan.save_weights('./weights/weights')
    print(f"Restored weights from {latest_ckpt}.")

    # Translate images
    out_dir = create_dir(args.output)
    for i, (imageA, imageB) in enumerate(input_images):
        predictionB = dclgan.netG_AB(imageA)[0].numpy()
        predictionB = (predictionB * 127.5 + 127.5).astype(np.uint8)
        imageA = (imageA[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        predictionA = dclgan.netG_BA(imageB)[0].numpy()
        predictionA = (predictionA * 127.5 + 127.5).astype(np.uint8)
        imageB = (imageB[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        _, ax = plt.subplots(1, 4, figsize=(20, 10))
        ax[0].imshow(imageA)
        ax[1].imshow(predictionB)
        ax[2].imshow(imageB)
        ax[3].imshow(predictionA)
        ax[0].set_title("InputA")
        ax[1].set_title("TranslatedB")
        ax[2].set_title("InputB")
        ax[3].set_title("TranslatedA")
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")
        ax[3].axis("off")

        plt.savefig(f'{out_dir}/infer={i + 1}.png')
        plt.close()


if __name__ == '__main__':
    main(ArgParse())
