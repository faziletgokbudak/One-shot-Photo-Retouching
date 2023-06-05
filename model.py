import functools

import tensorflow as tf
from tensorflow import keras
from keras import activations
from keras import layers, initializers

from utils import positional_encoding

from options import Options

args = Options().parse()


Dense = functools.partial(layers.Dense, kernel_initializer='uniform',
                          bias_initializer=initializers.constant(0.),
                          activation=activations.leaky_relu)

LastDense = functools.partial(layers.Dense, kernel_initializer='uniform',
                          bias_initializer=initializers.constant(0.),
                          activation=activations.softmax)


class CoeffMLP(keras.Model):

    def __init__(self, patch_size, A_kernels):
        super(CoeffMLP, self).__init__()
        self.A = A_kernels
        self.patch_size = patch_size
        self.dense1 = Dense(32, name='filters_1')
        self.dense2 = Dense(32, name='filters_2')
        self.dense3 = Dense(32, name='filters_3')
        self.last_layer = LastDense(args.num_matrices, name='filters_last')

    def call(self, inputs):
        y = self.dense1(inputs)
        y = self.dense2(y)
        y = self.dense3(y)
        f = self.last_layer(y)

        f_scalars_T = tf.cast(tf.transpose(f), tf.float32)
        A_dot_x_vectorized = tf.tensordot(self.A, tf.cast(tf.transpose(inputs), tf.float32), axes=1)
        f_repeated_vectorized = tf.repeat(tf.expand_dims(f_scalars_T, axis=1), 9, axis=1, name=None)
        weight_mult = A_dot_x_vectorized * f_repeated_vectorized
        y = tf.transpose(tf.math.reduce_sum(weight_mult, axis=0))

        return y


class MapMLP(keras.Model):

    def __init__(self, patch_size):
        super(MapMLP, self).__init__()
        self.patch_size = patch_size
        self.dense1 = Dense(32, name='filters_1')
        self.dense2 = Dense(32, name='filters_2')
        self.dense3 = Dense(32, name='filters_3')
        self.last_layer = LastDense(patch_size * 2, name='filters_last')

    def call(self, inputs):
        encoded_inputs = positional_encoding(inputs, 100)
        y = self.dense1(encoded_inputs)
        y = self.dense2(y)
        y = self.dense3(y)
        y_mod = self.last_layer(y)
        a = y_mod[:, :self.patch_size]
        b = y_mod[:, self.patch_size:]
        sum = tf.math.add(tf.math.multiply(inputs, a), b)
        return a, b, sum


class PositionalMLP(keras.Model):

    def __init__(self, patch_size, num_frequencies):
        super(PositionalMLP, self).__init__()
        self.freq_num = num_frequencies
        self.patch_size = patch_size
        self.dense1 = Dense(32, name='filters_1')
        self.dense2 = Dense(32, name='filters_2')
        self.dense3 = Dense(32, name='filters_3')
        self.last_layer = LastDense(patch_size, name='filters_last')

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float64, name=None)
        encoded_inputs = positional_encoding(inputs, self.freq_num)
        y = self.dense1(encoded_inputs)
        y = self.dense2(y)
        y = self.dense3(y)
        y = self.last_layer(y)
        y_mod = tf.cast(y, dtype=tf.float64)
        return y_mod


class ResidualMLP(keras.Model):

    def __init__(self, freq_num, num_channels, laplacian_levels=5):
        super(ResidualMLP, self).__init__()
        self.freq_num = freq_num
        self.dense1 = Dense(64, name='filters_1')
        self.dense2 = Dense(64 - laplacian_levels * num_channels, name='filters_2')
        self.dense3 = Dense(64, name='filters_3')
        self.last_layer = LastDense(laplacian_levels * num_channels, name='filters_last') #laplacian_level for train_concatenated

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float64, name=None)
        encoded_inputs = positional_encoding(inputs, self.freq_num)
        y = self.dense1(encoded_inputs)
        z = self.dense2(y)
        z_and_input = tf.keras.layers.Concatenate()([encoded_inputs, z])
        t = self.dense3(z_and_input)
        o = self.last_layer(t)
        return o


class MLP(keras.Model):

    def __init__(self, freq_num, num_channels, laplacian_levels=5):
        super(MLP, self).__init__()
        self.freq_num = freq_num
        self.dense1 = Dense(32, name='filters_1')
        self.dense2 = Dense(32, name='filters_2')
        self.dense3 = Dense(32, name='filters_3')
        self.last_layer = LastDense(laplacian_levels * num_channels, name='filters_last') #laplacian_level for train_concatenated

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float64, name=None)
        encoded_inputs = positional_encoding(inputs, self.freq_num)
        y = self.dense1(encoded_inputs)
        z = self.dense2(y)
        t = self.dense3(z)
        o = self.last_layer(t)

        return o
