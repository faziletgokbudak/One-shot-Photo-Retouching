import cv2
import numpy as np
import tensorflow as tf

from tfg import pyramid

from options import Options
from utils import positional_encoding, convert_rgb_to_y, crop_center, \
    calculate_psnr, calculate_ssim, crop_center_test, apply_detail_decomposition, iterative_downsample

args = Options().parse()


def image_preprocess(img_path, suffix, pyramid_type):
    img = cv2.imread(img_path)
    if suffix == 'input' or 'output':
        img = crop_center(img, 100, 100)
    else:
        img = crop_center_test(img, 100, 100)

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    img_y_np = img_ycrcb[:, :, 0] / 255.
    img_cr_np = img_ycrcb[:, :, 1] / 255.
    img_cb_np = img_ycrcb[:, :, 2] / 255.

    # Y Channel
    img_y = tf.dtypes.cast(tf.convert_to_tensor(img_y_np), dtype=tf.float32)
    tensor_img_y = tf.expand_dims(img_y, axis=0)  # extra dim for batches
    tensor_img_y = tf.expand_dims(tensor_img_y, axis=-1)  # extra dim for color

    # Cr channel
    img_cr = tf.dtypes.cast(tf.convert_to_tensor(img_cr_np), dtype=tf.float32)
    tensor_img_cr = tf.expand_dims(img_cr, axis=0)  # extra dim for batches
    tensor_img_cr = tf.expand_dims(tensor_img_cr, axis=-1)  # extra dim for color

    # Cb channel
    img_cb = tf.dtypes.cast(tf.convert_to_tensor(img_cb_np), dtype=tf.float32)
    tensor_img_cb = tf.expand_dims(img_cb, axis=0)  # extra dim for batches
    tensor_img_cb = tf.expand_dims(tensor_img_cb, axis=-1)  # extra dim for color

    if pyramid_type == 'laplacian':
        laplacian_img_y = pyramid.split(tensor_img_y, args.laplacian_level, name='input')  # creates Laplacian pyramid
        return laplacian_img_y, 0, tensor_img_cr, tensor_img_cb

    elif pyramid_type == 'guided':
        img_y_np = np.asarray(img_y_np, dtype=np.float32)
        guided_img_y, base = apply_detail_decomposition(img_y_np)
        downsampled_detail_bands = []
        for i, band in enumerate(guided_img_y):
            downsampled_detail_bands.append(iterative_downsample(band, i, 0))
        return downsampled_detail_bands, base, tensor_img_cr, tensor_img_cb