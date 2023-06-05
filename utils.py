import cv2
import math
import numpy as np

import tensorflow as tf

from typing import Union, Sequence, Tuple
from sklearn.feature_extraction import image as extraction

# from cv2.ximgproc import guidedFilter

from options import Options

Integer = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64]
Float = Union[float, np.float16, np.float32, np.float64]
TensorLike = Union[Integer, Float, Sequence, np.ndarray, tf.Tensor, tf.Variable]

args = Options().parse()


def extract_patches_from_laplace_layer(laplace_layer):
    laplace_np = np.array(tf.squeeze(tf.squeeze(laplace_layer, axis=0), axis=-1))
    patches_np = np.squeeze(extraction.extract_patches_2d(laplace_np,
                                                          (args.patch_size[0] * args.patch_size[1], 1)), axis=-1)
    patches = tf.convert_to_tensor(patches_np)
    return patches


def reconstruct_image_from_subbands(subbands: TensorLike, base: TensorLike) -> TensorLike:
    img = tf.zeros(shape=subbands[:, :, 0].shape)
    for i in range(subbands.shape[-1]):
        img = tf.add(img, subbands[:, :, i])
    return tf.add(img, base)


def extract_image_patches(image: TensorLike,
                          patch_dims: TensorLike,
                          stride: TensorLike) -> tf.Tensor:
    """Extracts the image patches of dimension, patch_dims,
   with a stride of integer number.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    patch_dims: A tensor of shape `[H_p, W_p]`, where `H_p` and `W_p` are the
      height and width of the patches.
    stride: An integer number that indicates the rowwise movement between
    successive patches

  Returns:
    A tensor of shape `[N, H_p * W_p]`, where `N` is the total number of patches.

  """

    patches = tf.image.extract_patches(images=image,
                                       sizes=[1, patch_dims[0], patch_dims[1], 1],
                                       strides=[1, stride, stride, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')

    squeezed_patches = tf.squeeze(patches, axis=0)

    reshaped_squeezed_patches = tf.reshape(squeezed_patches,
                                           [squeezed_patches.shape[0] *
                                            squeezed_patches.shape[1],
                                            patch_dims[0] * patch_dims[1]])
    return reshaped_squeezed_patches


def reconstruct_weight_image(laplacian_level,
                                   channel,
                                   patches: TensorLike,
                                   img_dims: TensorLike,
                                   stride: TensorLike,
                                   dtype=np.float32) -> tf.Tensor:

    patches = tf.reshape(patches, [patches.shape[0], 1, 1])
    new_image = np.zeros([img_dims[0], img_dims[1]], dtype=dtype)
    # merging_map = np.zeros([img_dims[0], img_dims[1]], dtype=dtype)

    count = 0
    for i in range(0, img_dims[0] - 3 + 1, stride):
        for j in range(0, img_dims[1] - 3 + 1, stride):
            new_image[i, j] = new_image[i, j] + patches[count][:][:]
            count += 1
    return new_image


def reconstruct_image_from_patches(laplacian_level,
                                   channel,
                                   patches: TensorLike,
                                   img_dims: TensorLike,
                                   stride: TensorLike,
                                   dtype=np.float32) -> tf.Tensor:
    """Combines the patches and saves the reconstructed image.

  Args:
    patches: A tensor of shape `[N, H_p, W_p]`, where `N` is the total number
     of patches, `H_p` and `W_p` are the height and width of the patches.
    img_dims: A tensor of shape `[H, W]`, where `H` and `W` are the
      height and width of the reconstructed image.
    stride: An integer number that indicates the rowwise movement between
    successive patches.

  Returns:
    A tensor of shape `[H, W]`, which is the reconstructed image.
    :param dtype:
    :param stride:
    :param img_dims:
    :param patches:
    :param laplacian_level:
    :param tree_level:

  """
    patches = tf.reshape(patches, [patches.shape[0], 3, 3])
    new_image = np.zeros([img_dims[0], img_dims[1]], dtype=dtype)
    merging_map = np.zeros([img_dims[0], img_dims[1]], dtype=dtype)

    count = 0
    patch_dims = [patches.shape[1], patches.shape[2]]
    for i in range(0, img_dims[0] - patch_dims[0] + 1, stride):
        for j in range(0, img_dims[1] - patch_dims[1] + 1, stride):
            new_image[i:i + patch_dims[0], j:j + patch_dims[1]] = new_image[i:i + patch_dims[0], j:j + patch_dims[1]] + \
                                                                  patches[count][:][:]
            merging_map[i:i + patch_dims[0], j:j + patch_dims[1]] += 1
            count += 1

    image_new = np.divide(new_image, merging_map)

    return image_new


def positional_encoding(features: tf.Tensor,
                        num_frequencies: int,
                        name="positional_encoding") -> tf.Tensor:
  """taken from Tensorflow Graphics Library
  Positional enconding of a tensor as described in the NeRF paper (https://arxiv.org/abs/2003.08934).
  Args:
    features: A tensor of shape `[A1, ..., An, M]` where M is the dimension
       of the features.
    num_frequencies: Number N of frequencies for the positional encoding.
    name: A name for this op that defaults to "positional_encoding".
  Returns:
    A tensor of shape `[A1, ..., An, 2*N*M + M]`.
  """
  with tf.name_scope(name):
    features = tf.convert_to_tensor(value=features)
    output = [features]
    for i in range(num_frequencies):
      for fn in [tf.sin, tf.cos]:
        output.append(fn(2. ** i * math.pi * features))
    return tf.concat(output, -1)


def convert_ycbcr_to_rgb(img):
    r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
    g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
    b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def convert_rgb_to_y(img):
    if np.amax(img) <= 1.:
        img = img * 255.
    return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.


def convert_rgb_to_ycbcr(img):
    y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
    cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    # return y, cb, cr
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def crop_center(img,cropx,cropy):
    if len(img.shape) == 3:
        y,x = img.shape[0], img.shape[1]
        startx = x//2-(cropx//2) - 150
        starty = y//2-(cropy//2) - 100
        return img[starty:starty+cropy, startx:startx+cropx, :]
    else:
        y,x = img.shape
        startx = x//2-(cropx//2) - 150
        starty = y//2-(cropy//2) - 100
        return img[starty:starty+cropy, startx:startx+cropx]


def crop_center_test(img,cropx,cropy):
    if len(img.shape) == 3:
        y,x = img.shape[0], img.shape[1]
        startx = x//2-(cropx//2) - 150
        starty = y//2-(cropy//2) - 150
        return img[starty:starty+cropy, startx:startx+cropx, :]
    else:
        y,x = img.shape
        startx = x//2-(cropx//2) - 150
        starty = y//2-(cropy//2) - 150
        return img[starty:starty+cropy, startx:startx+cropx]


def make_variables(k, initializer):
    return tf.Variable(initializer(shape=[k], dtype=tf.float32))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def apply_detail_decomposition(img: TensorLike) -> Tuple[list, TensorLike]:
    N = math.floor(math.log2(min(img.shape[0], img.shape[1])))  # number of layers
    r = 3
    eps = 0.25 * 0.25
    tmp1, p = img, img
    subbands = []
    for i in range(1, N):
        tmp2 = tmp1
        # tmp1 = guidedFilter(tmp2, p, r ** i, eps)  # double the spatial extent each time
        layer = tmp2 - tmp1
        layer = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(layer), axis=0), axis=-1)
        subbands.append(layer)  # a subband is the difference of two succcessively filtered versions of the image
    residual = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(tmp1), axis=0), axis=-1)
    return subbands, residual


def reconstruct_image_from_subbands_tf(subbands: TensorLike, base: TensorLike) -> TensorLike:
    img = tf.zeros(shape=subbands[0].shape)
    for i in range(len(subbands)):
        img += subbands[i]
    return img + base


def downsample(image: TensorLike,
                kernel: TensorLike) -> tf.Tensor:
    """Downsamples the image using a convolution with stride 2.
  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    kernel: A tensor of shape `[H_k, W_k, C, C]`, where `H_k` and `W_k` are the
      height and width of the kernel.
  Returns:
    A tensor of shape `[B, H_d, W_d, C]`, where `H_d` and `W_d` are the height
    and width of the downsampled image.
  """
    return tf.nn.conv2d(
        input=image, filters=kernel, strides=[1, 2, 2, 1], padding="SAME")


def binomial_kernel(num_channels: int,
                     dtype: tf.DType = tf.float32) -> tf.Tensor:
    """Creates a 5x5 binomial kernel.
  Args:
    num_channels: The number of channels of the image to filter.
    dtype: The type of an element in the kernel.
  Returns:
    A tensor of shape `[5, 5, num_channels, num_channels]`.
  """
    kernel = np.array((1., 4., 6., 4., 1.), dtype=dtype.as_numpy_dtype)
    kernel = np.outer(kernel, kernel)
    kernel /= np.sum(kernel)
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    return tf.constant(kernel, dtype=dtype) * tf.eye(num_channels, dtype=dtype)


def iterative_downsample(img, i, count):
    if count == i:
        return img
    else:
        kernel = binomial_kernel(1)
        img = downsample(img, kernel)
        return iterative_downsample(img, i, count+1)