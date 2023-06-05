import os
import numpy as np
import tensorflow as tf

from options import Options
from utils import extract_patches
from train_loop import train_loop
from dataloader import image_preprocess


args = Options().parse()


if __name__ == '__main__':

    # Preprocessing of images
    laplacian_input_y, base_input, tensor_input_cr, tensor_input_cb = image_preprocess(args.input_path, 'input', 'laplacian')
    laplacian_output_y, base_output, tensor_output_cr, tensor_output_cb = image_preprocess(args.output_path, 'output', 'laplacian')

    model_path_kernels = args.model_path + '/A_kernels'
    if not os.path.exists(model_path_kernels):
        os.makedirs(model_path_kernels)

    mixed_laplacian = []

    # Y channel - Laplacian layers' training
    for i in range(args.laplacian_level):

        input_patches = extract_patches(laplacian_input_y[i])
        output_patches = extract_patches(laplacian_output_y[i])

        model, A_kernel = train_loop(input_patches, output_patches)

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        np.save(model_path_kernels + '/y_L{}'.format(i), A_kernel.numpy())
        model_path = args.model_path + '/y_L{}'.format(i)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save(model_path)

    if args.chrom:
        # Cr channel training
        input_patches_cr = extract_patches(tensor_input_cr)
        output_patches_cr = extract_patches(tensor_output_cr)

        model_cr, A_cr = train_loop(input_patches_cr, output_patches_cr)
        np.save(model_path_kernels + '/cr', A_cr.numpy())

        model_path_cr = args.model_path + '/cr'
        if not os.path.exists(model_path_cr):
            os.makedirs(model_path_cr)
        model_cr.save(model_path_cr)

        # Cb channel training
        input_patches_cb = extract_patches(tensor_input_cb)
        output_patches_cb = extract_patches(tensor_output_cb)

        model_cb, A_cb = train_loop(input_patches_cb, output_patches_cb)
        np.save(model_path_kernels + '/cb', A_cb.numpy())

        model_path_cb = args.model_path + '/cb'
        if not os.path.exists(model_path_cb):
            os.makedirs(model_path_cb)
        model_cb.save(model_path_cb)



