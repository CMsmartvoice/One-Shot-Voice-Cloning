import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.

    Args:
        initializer_range: float, initializer range for stddev.

    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.

    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

class MaskInstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, layer_norm_eps, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm_eps = layer_norm_eps

    def _cal_mean_std(self, inputs, mask):
        expend_mask = tf.cast(tf.expand_dims(mask, axis=2), inputs.dtype)
        sums        = tf.math.reduce_sum(tf.cast(mask, inputs.dtype), axis=-1, keepdims=True)

        mean = tf.math.reduce_sum(inputs * expend_mask, axis=1) / sums

        std = tf.math.sqrt(
                tf.math.reduce_sum(
                    tf.math.pow(inputs - tf.expand_dims(mean, 1), 2) * expend_mask, axis = 1
                    ) / sums + self.layer_norm_eps
                            )

        return mean, std, expend_mask

    def call(self, inputs, mask, return_mean_std=False):
        '''
        inputs: [B, T, hidden_size]
        mask:   [B, T]
        '''
        mean, std, expend_mask = self._cal_mean_std(inputs, mask)

        outputs = (inputs - tf.expand_dims(mean, 1)) / tf.expand_dims(std, 1) * expend_mask

        if return_mean_std:
            return outputs, mean, std
        else:
            return outputs

# TODO
class ConditionalNormalization(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.scale = tf.keras.layers.Dense(
            config.hidden_size,
            use_bias           = False,
            kernel_initializer = get_initializer(config.initializer_range),
            name               = "Scale",
        )

        self.mean = tf.keras.layers.Dense(
            config.hidden_size,
            use_bias           = False,
            kernel_initializer = get_initializer(config.initializer_range),
            name               = "Mean",
        )

        if config.conditional_norm_type == "Layer":
            self.norm_layer = tf.keras.layers.LayerNormalization(
                center  = False,
                scale   = False,
                epsilon = config.layer_norm_eps,
                name    = "LayerNorm",
            )
        elif config.conditional_norm_type == "Instance":
            # self.norm_layer = tfa.layers.InstanceNormalization(
            #     center  = False,
            #     scale   = False,
            #     epsilon = config.layer_norm_eps,
            #     name    = "InstanceNorm",
            # )
            self.norm_layer = MaskInstanceNormalization(config.layer_norm_eps)
        else:
            print(f"Not support norm type {config.conditional_norm_type} !")
            exit(0)

    def call(self, inputs, conds, mask):
        '''
        inputs: [B, T, hidden_size]
        conds:  [B, 1, C']
        mask:   [B, T]
        '''
        if self.config.conditional_norm_type == "Layer":
            tmp = self.norm_layer(inputs)
        elif self.config.conditional_norm_type == "Instance":
            tmp = self.norm_layer(inputs, mask)

        scale = self.scale(conds)
        mean  = self.mean(conds)

        return tmp * scale + mean