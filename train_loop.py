import math

import tensorflow as tf
import tensorflow_probability as tfp

from model import CoeffMLP
from options import Options

args = Options().parse()
l1_loss = tf.keras.losses.MeanAbsoluteError()


def make_variables(k, initializer):
    return tf.Variable(initializer(shape=k, dtype=tf.float32))


def generate_random_filters():
    """Generate random filters m and n of both sides of tree for initialisation
    Filters are variables, that is, learnable."""
    patch_len = args.patch_size[0] * args.patch_size[1]
    kernel = make_variables([args.num_matrices, patch_len, patch_len], tf.random_uniform_initializer(minval=0., maxval=1.))
    return kernel


def calculate_row_variance(rows, row_order):
    set_number = 2 ** (row_order - 1)
    row_var = tf.reduce_sum(tfp.stats.variance(rows[:args.num_matrices // set_number], sample_axis=0))
    for i in range(1, set_number):
        row_var = row_var + tf.reduce_sum(tfp.stats.variance(rows[i*(args.num_matrices // set_number):(i+1)*(args.num_matrices // set_number)], sample_axis=0))
    return row_var


def calculate_total_variance(A):
    total_var = 0.
    row_number = int(math.log2(args.num_matrices))
    for k in range(1, row_number + 1):
        total_var += calculate_row_variance(
            tf.convert_to_tensor([A[i, :, :][k-1, :] for i in range(args.num_matrices)]), k)
    return total_var


def train_loop(input_patches, output_patches):
    A = generate_random_filters()  # transformation matrices
    model = CoeffMLP(patch_size=args.patch_size[0] * args.patch_size[1], A_kernels=A)  # patch-adaptive weights

    lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr, args.epoch, 0.96, staircase=False, name=None
    )
    l1_loss = tf.keras.losses.MeanAbsoluteError()
    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)

    model.compile(optimizer=opt, loss=[l1_loss])

    history = model.fit(
        input_patches,
        output_patches,
        batch_size=args.batch_size,
        epochs=args.epoch,
    )
    return model, A
