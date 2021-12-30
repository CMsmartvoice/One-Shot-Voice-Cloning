import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_tts.models.moduls.conditional import MaskInstanceNormalization

def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.

    Args:
        initializer_range: float, initializer range for stddev.

    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.

    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

class ConvModul(tf.keras.layers.Layer):
    def __init__(self, hidden_size, kernel_size, initializer_range, layer_norm_eps=1e-5, **kwargs):
        super().__init__(**kwargs)

        self.conv_0 = tf.keras.layers.Conv1D(
            filters            = hidden_size,
            kernel_size        = kernel_size,
            kernel_initializer = get_initializer(initializer_range),
            padding            = 'same',
        )

        self.conv_1 = tf.keras.layers.Conv1D(
            filters            = hidden_size,
            kernel_size        = kernel_size,
            kernel_initializer = get_initializer(initializer_range),
            padding            = 'same',
        )

        self.atc = tf.keras.layers.Activation(tf.nn.relu)

        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=layer_norm_eps) # TODO

    def call(self, x):
        y = self.conv_0(x)
        y = self.batch_norm(y)
        y = self.atc(y)
        y = self.conv_1(y)
        return y

class EncConvBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.conv = ConvModul(
            config.adain_filter_size,
            config.enc_kernel_size,
            config.initializer_range,
            config.layer_norm_eps)

    def call(self, x):
        return x + self.conv(x)

class DecConvBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dec_conv = ConvModul(
            config.adain_filter_size,
            config.dec_kernel_size,
            config.initializer_range,
            config.layer_norm_eps)

        self.gen_conv = ConvModul(
            config.adain_filter_size,
            config.gen_kernel_size,
            config.initializer_range,
            config.layer_norm_eps)

    def call(self, x):
        y = self.dec_conv(x)
        y = y + self.gen_conv(y)
        return x + y

class AadINEncoder(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config          = config
        self.in_hidden_size  = config.adain_filter_size # 256
        self.out_hidden_size = config.content_latent_dim # content_latent_dim
        self.n_conv_blocks   = config.n_conv_blocks

        self.in_conv =  tf.keras.layers.Conv1D(
            filters            = self.in_hidden_size,
            kernel_size        = 1,
            kernel_initializer = get_initializer(config.initializer_range),
            padding            = 'same',
        )

        self.out_conv =  tf.keras.layers.Conv1D(
            filters            = self.out_hidden_size,
            kernel_size        = 1,
            kernel_initializer = get_initializer(config.initializer_range),
            padding            = 'same',
        )

        self.inorm = MaskInstanceNormalization(config.layer_norm_eps)

        self.conv_blocks = [
            EncConvBlock(config) for _ in range(self.n_conv_blocks)
        ]

    def call(self, x, mask):
        means = []
        stds  = []

        y = self.in_conv(x) # 80 -> 256

        for block in self.conv_blocks:
            y = block(y)
            y, mean, std = self.inorm(y, mask, return_mean_std=True)
            means.append(mean)
            stds.append(std)

        y = self.out_conv(y) # 256 -> 128 + 4

        # TODO sigmoid

        means.reverse()
        stds.reverse()

        return y, means, stds


class AdaINDecoder(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.in_hidden_size  = config.adain_filter_size # 256
        self.out_hidden_size = config.num_mels # 80
        self.n_conv_blocks   = config.n_conv_blocks

        self.in_conv =  tf.keras.layers.Conv1D(
            filters            = self.in_hidden_size,
            kernel_size        = 1,
            kernel_initializer = get_initializer(config.initializer_range),
            padding            = 'same',
        )

        self.out_conv =  tf.keras.layers.Conv1D(
            filters            = self.out_hidden_size,
            kernel_size        = 1,
            kernel_initializer = get_initializer(config.initializer_range),
            padding            = 'same',
        )

        self.inorm = MaskInstanceNormalization(config.layer_norm_eps)

        self.conv_blocks = [
            DecConvBlock(config) for _ in range(self.n_conv_blocks)
        ]

    def call(self, enc, cond, mask):
        _, means, stds = cond
 
        # TODO
        # y, means, stds = cond
        # _, mean, std = self.inorm(y, mask, return_mean_std=True)
        # enc = self.inorm(enc, mask)
        # enc = enc * tf.expand_dims(std, 1) + tf.expand_dims(mean, 1)

        y = self.in_conv(enc) # 132 -> 256

        for block, mean, std in zip(self.conv_blocks, means, stds):
            y = self.inorm(y, mask)
            y = y * tf.expand_dims(std, 1) + tf.expand_dims(mean, 1)
            y = block(y)

        y = self.out_conv(y) # 256 -> 80

        return y
