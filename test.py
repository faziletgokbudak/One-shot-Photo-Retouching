import os
import time

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tfg import pyramid
from sklearn.feature_extraction import image as extraction

from options import Options
from dataloader import image_preprocess

from utils import extract_patches

args = Options().parse()

if __name__ == '__main__':
    t = time.time()

    laplacian_test_y, base, tensor_test_cr, tensor_test_cb = image_preprocess(args.test_path, 'input', 'laplacian')

    img_width = tensor_test_cr.shape[1]
    img_height = tensor_test_cr.shape[2]

    mixed_laplacian = []

    for i in range(args.laplacian_level):
        test_patches = extract_patches(laplacian_test_y[i])

        band_width = laplacian_test_y[i].shape[1]
        band_height = laplacian_test_y[i].shape[2]

        # A_kernels = tf.convert_to_tensor(np.load(args.model_path + '/A_kernels/y_L{}.npy'.format(i)))

        model_path = args.model_path + '/y_L{}'.format(i)
        model = tf.keras.models.load_model(model_path)

        Y_pred = model.predict(
            test_patches,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )

        Y_pred_np = np.array(tf.expand_dims(Y_pred, axis=-1))
        test_output_2d = extraction.reconstruct_from_patches_2d(Y_pred_np, [band_width, band_height])
        test_output_2d = tf.convert_to_tensor(test_output_2d, dtype=tf.float32)

        test_output = tf.expand_dims(tf.expand_dims(test_output_2d, axis=0), axis=-1)
        mixed_laplacian.append(test_output)

    mixed_laplacian.append(laplacian_test_y[-1])
    img = tf.squeeze(tf.squeeze(pyramid.merge(mixed_laplacian, name='None'), axis=0), axis=-1)  # reconstructs image from pyramid

    img_y_np = np.array(img) * 255.

    # In case chrominance channels are learned
    if args.chrom:
        test_patches_cr = extract_patches(tensor_test_cr)
        test_patches_cb = extract_patches(tensor_test_cb)

        model_path_cr = args.model_path + '/cr'
        model_cr = tf.keras.models.load_model(model_path_cr)

        model_path_cb = args.model_path + '/cb'
        model_cb = tf.keras.models.load_model(model_path_cb)

        Cr_pred = model_cr.predict(
            test_patches_cr,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )

        Cb_pred = model_cb.predict(
            test_patches_cb,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )

        Cr_pred_np = np.array(tf.expand_dims(Cr_pred, axis=-1))
        test_output_2d_cr = extraction.reconstruct_from_patches_2d(Cr_pred_np, [img_width, img_height]) * 255.

        Cb_pred_np = np.array(tf.expand_dims(Cb_pred, axis=-1))
        test_output_2d_cb = extraction.reconstruct_from_patches_2d(Cb_pred_np, [img_width, img_height]) * 255.

    else:
        test_output_2d_cr = np.array(np.squeeze(np.squeeze(tensor_test_cr, axis=0), axis=-1)) * 255.
        test_output_2d_cb = np.array(np.squeeze(np.squeeze(tensor_test_cb, axis=0), axis=-1)) * 255.

    # save each channel, then combine them
    cv2.imwrite(os.getcwd() + '/y_only.png', img_y_np)
    cv2.imwrite(os.getcwd() + '/cr_only.png', test_output_2d_cr)
    cv2.imwrite(os.getcwd() + '/cb_only.png', test_output_2d_cb)

    reconstructed_y = cv2.imread(os.getcwd() + '/y_only.png')[:, :, 0]
    reconstructed_cr = cv2.imread(os.getcwd() + '/cr_only.png')[:, :, 0]
    reconstructed_cb = cv2.imread(os.getcwd() + '/cb_only.png')[:, :, 0]

    reconstructed = np.dstack((reconstructed_y, reconstructed_cr, reconstructed_cb))
    bgr = cv2.cvtColor(reconstructed, cv2.COLOR_YCR_CB2BGR)

    cv2.imwrite(args.test_output_path, bgr)