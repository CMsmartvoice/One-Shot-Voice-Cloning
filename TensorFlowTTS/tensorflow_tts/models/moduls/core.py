# -*- coding: utf-8 -*-
# Copyright 2020 The FastSpeech Authors, The HuggingFace Inc. team and Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensorflow Model modules for FastSpeech."""

import numpy as np
import tensorflow as tf
import scipy.stats


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.

    Args:
        initializer_range: float, initializer range for stddev.

    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.

    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def gelu(x):
    """Gaussian Error Linear unit."""
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def gelu_new(x):
    """Smoother gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    """Swish activation function."""
    return x * tf.sigmoid(x)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


ACT2FN = {
    "identity": tf.keras.layers.Activation("linear"),
    "tanh": tf.keras.layers.Activation("tanh"),
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
    "mish": tf.keras.layers.Activation(mish),
}


class TFEmbedding(tf.keras.layers.Embedding):
    """Faster version of embedding."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.int32)
        outputs = tf.gather(self.embeddings, inputs)
        return outputs


class TFFastSpeechEmbeddings(tf.keras.layers.Layer):
    """Construct charactor/phoneme/positional/speaker embeddings."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.vocab_size        = config.vocab_size
        self.hidden_size       = config.encoder_self_attention_params.hidden_size
        self.initializer_range = config.initializer_range
        self.config            = config

    def build(self, input_shape):
        """Build shared charactor/phoneme embedding layers."""
        with tf.name_scope("charactor_embeddings"):
            self.charactor_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(self, input_ids):
        return tf.gather(self.charactor_embeddings, input_ids)


class TFFastSpeechSelfAttention(tf.keras.layers.Layer):
    """Self attention module for fastspeech."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * config.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.config = config

        # TODO
        # self.half_win = config.local_attention_halfwin_size
        # self.frames_max = 100
        # self.local_maxs = self._local_attention_mask()
        # self.local_ones = tf.ones([self.frames_max, self.frames_max], tf.float32)

    def transpose_for_scores(self, x, batch_size):
        """Transpose to calculate attention scores."""
        x = tf.reshape(
            x,
            (batch_size, -1, self.num_attention_heads, self.config.attention_head_size),
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # def _local_attention_mask(self, frames_num):
    #     xv, yv = tf.meshgrid(tf.range(frames_num), tf.range(frames_num), indexing="ij")
    #     f32_matrix = tf.cast(yv - xv, tf.float32)

    #     val = f32_matrix[0][self.half_win]

    #     local1 = tf.math.greater_equal(f32_matrix, -val)
    #     local2 = tf.math.less_equal(f32_matrix, val)
        
    #     return tf.cast(tf.logical_and(local1, local2), tf.float32)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        batch_size = tf.shape(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(tf.shape(key_layer)[-1], attention_scores.dtype)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # extended_attention_masks for self attention encoder.
            extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            extended_attention_mask = tf.cast(extended_attention_mask, attention_scores.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            attention_scores = attention_scores + extended_attention_mask

            # TODO
            # frames_num = tf.shape(attention_mask)[-1]
            # local_attention_mask = tf.cond(tf.greater(frames_num, self.half_win + 1),
            #                                 lambda: self._local_attention_mask(frames_num),
            #                                 lambda: tf.ones([frames_num, frames_num], tf.float32))
            # local_attention_mask = (1.0 - local_attention_mask) * -1e9
            # attention_scores = attention_scores + local_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class TFFastSpeechSelfOutput(tf.keras.layers.Layer):
    """Fastspeech output of self attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFFastSpeechAttention(tf.keras.layers.Layer):
    """Fastspeech attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.self_attention = TFFastSpeechSelfAttention(config, name="self")
        self.dense_output = TFFastSpeechSelfOutput(config, name="output")

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs

        self_outputs = self.self_attention(
            [input_tensor, attention_mask], training=training
        )
        attention_output = self.dense_output(
            [self_outputs[0], input_tensor], training=training
        )
        masked_attention_output = attention_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=attention_output.dtype
        )
        outputs = (masked_attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class TFFastSpeechIntermediate(tf.keras.layers.Layer):
    """Intermediate representation module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv1d_1 = tf.keras.layers.Conv1D(
            config.intermediate_size,
            kernel_size=config.intermediate_kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding="same",
            name="conv1d_1",
        )
        self.conv1d_2 = tf.keras.layers.Conv1D(
            config.hidden_size,
            kernel_size=config.intermediate_kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding="same",
            name="conv1d_2",
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, inputs):
        """Call logic."""
        hidden_states, attention_mask = inputs

        hidden_states = self.conv1d_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.conv1d_2(hidden_states)

        masked_hidden_states = hidden_states * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=hidden_states.dtype
        )
        return masked_hidden_states


class TFFastSpeechOutput(tf.keras.layers.Layer):
    """Output module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, input_tensor = inputs

        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFFastSpeechLayer(tf.keras.layers.Layer):
    """Fastspeech module (FFT module on the paper)."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.attention = TFFastSpeechAttention(config, name="attention")
        self.intermediate = TFFastSpeechIntermediate(config, name="intermediate")
        self.bert_output = TFFastSpeechOutput(config, name="output")

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        attention_outputs = self.attention(
            [hidden_states, attention_mask], training=training
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(
            [attention_output, attention_mask], training=training
        )
        layer_output = self.bert_output(
            [intermediate_output, attention_output], training=training
        )
        masked_layer_output = layer_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=layer_output.dtype
        )
        outputs = (masked_layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class TFFastSpeechEncoder(tf.keras.layers.Layer):
    """Fast Speech encoder module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [
            TFFastSpeechLayer(config, name="layer_._{}".format(i))
            for i in range(config.num_hidden_layers)
        ]

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        all_hidden_states = ()
        all_attentions = ()
        for _, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                [hidden_states, attention_mask], training=training
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class TFFastSpeechDecoder(TFFastSpeechEncoder):
    """Fast Speech decoder module."""

    def __init__(self, config, **kwargs):
        self.is_compatible_encoder = kwargs.pop("is_compatible_encoder", True)

        super().__init__(config, **kwargs)
        self.config = config

        if self.is_compatible_encoder is False:
            self.project_compatible_decoder = tf.keras.layers.Dense(
                units=config.hidden_size, name="project_compatible_decoder"
            )

    def call(self, inputs, training=False):
        hidden_states, encoder_mask = inputs

        if self.is_compatible_encoder is False:
            hidden_states = self.project_compatible_decoder(hidden_states)

        return super().call([hidden_states, encoder_mask], training=training)


class TFTacotronPostnet(tf.keras.layers.Layer):
    """Tacotron-2 postnet."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_postnet):
            conv = tf.keras.layers.Conv1D(
                filters=config.postnet_conv_filters
                if i < config.n_conv_postnet - 1
                else config.num_mels,
                kernel_size=config.postnet_conv_kernel_sizes,
                padding="same",
                name="conv_._{}".format(i),
            )
            batch_norm = tf.keras.layers.BatchNormalization(
                axis=-1, name="batch_norm_._{}".format(i)
            )
            self.conv_batch_norm.append((conv, batch_norm))
        self.dropout = tf.keras.layers.Dropout(
            rate=config.postnet_dropout_rate, name="dropout"
        )
        self.activation = [tf.nn.tanh] * (config.n_conv_postnet - 1) + [tf.identity]

    def call(self, inputs, training=False):
        """Call logic."""
        outputs, mask = inputs
        extended_mask = tf.cast(tf.expand_dims(mask, axis=2), outputs.dtype)
        for i, (conv, bn) in enumerate(self.conv_batch_norm):
            outputs = conv(outputs)
            outputs = bn(outputs)
            outputs = self.activation[i](outputs)
            outputs = self.dropout(outputs, training=training)
        return outputs * extended_mask

# TODO Drop infer trainning=False
class TFFastSpeechVariantPredictor(tf.keras.layers.Layer):
    """FastSpeech variant predictor module."""

    def __init__(self, config, sub_name="f0", is_sigmod=False, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.is_sigmod = is_sigmod
        self.conv_layers = []
        for i in range(config.num_variant_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    config.variant_predictor_filters,
                    config.variant_predictor_kernel_sizes,
                    padding="same",
                    name="{}_conv_._{}".format(sub_name, i),
                )
            )

            self.conv_layers.append(tf.keras.layers.Activation(tf.nn.relu))

            self.conv_layers.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=config.layer_norm_eps, name="{}_LayerNorm_._{}".format(sub_name, i)
                )
            )

            self.conv_layers.append(
                tf.keras.layers.Dropout(config.variant_predictor_dropout_probs)
            )
        self.conv_layers_sequence = tf.keras.Sequential(self.conv_layers, name=sub_name)
        self.output_layer = tf.keras.layers.Dense(1)

        if self.is_sigmod:
            self.sigmod_layer = tf.keras.layers.Activation(tf.nn.sigmoid)

    def call(self, inputs, training=False):
        """Call logic."""
        encoder_hidden_states, attention_mask = inputs
        attention_mask = tf.cast(tf.expand_dims(attention_mask, 2), encoder_hidden_states.dtype)

        # mask encoder hidden states
        masked_encoder_hidden_states = encoder_hidden_states * attention_mask

        # pass though first layer
        outputs = self.conv_layers_sequence(masked_encoder_hidden_states)
        outputs = self.output_layer(outputs)

        if self.is_sigmod:
            outputs = self.sigmod_layer(outputs)

        masked_outputs = outputs * attention_mask

        return tf.squeeze(masked_outputs, -1)

class TFFastSpeechDurationPredictor(tf.keras.layers.Layer):
    """FastSpeech duration predictor module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_layers = []
        for i in range(config.num_duration_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    config.duration_predictor_filters,
                    config.duration_predictor_kernel_sizes,
                    padding="same",
                    name="conv_._{}".format(i),
                )
            )

            self.conv_layers.append(tf.keras.layers.Activation(tf.nn.relu))

            self.conv_layers.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=config.layer_norm_eps, name="LayerNorm_._{}".format(i)
                )
            )

            self.conv_layers.append(
                tf.keras.layers.Dropout(config.duration_predictor_dropout_probs)
            )

        self.conv_layers_sequence = tf.keras.Sequential(self.conv_layers)

        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        """Call logic."""
        encoder_hidden_states, attention_mask = inputs
        attention_mask = tf.cast(tf.expand_dims(attention_mask, 2), encoder_hidden_states.dtype)

        # mask encoder hidden states
        masked_encoder_hidden_states = encoder_hidden_states * attention_mask

        # pass though first layer
        outputs = self.conv_layers_sequence(masked_encoder_hidden_states)
        outputs = self.output_layer(outputs)
        masked_outputs = outputs * attention_mask
        # return tf.squeeze(tf.nn.relu(masked_outputs), -1)  # make sure positive value.
        return tf.squeeze(masked_outputs, -1)

class TFFastSpeechLengthRegulator(tf.keras.layers.Layer):
    """FastSpeech lengthregulator module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        self.enable_tflite_convertible = kwargs.pop("enable_tflite_convertible", False)
        super().__init__(**kwargs)
        self.config = config

        self.addfeatures_num = 0

        if config.addfeatures_num > 0:
            self._compute_coarse_coding_features()
            self.addfeatures_num = config.addfeatures_num
            if config.isaddur:
                self.addfeatures_num += 1

    def _compute_coarse_coding_features(self):
        npoints = 600

        x1 = np.linspace(-1.5, 1.5, npoints)
        x2 = np.linspace(-1.0, 2.0, npoints)
        x3 = np.linspace(-0.5, 2.5, npoints)
        x4 = np.linspace(0.0, 3.0, npoints)

        mu1 = 0.0
        mu2 = 0.5
        mu3 = 1.0
        mu4 = 1.5

        sigma = 0.4

        self.cc_features0 = tf.convert_to_tensor(scipy.stats.norm.pdf(x1, mu1, sigma), tf.float32)
        self.cc_features1 = tf.convert_to_tensor(scipy.stats.norm.pdf(x2, mu2, sigma), tf.float32)
        self.cc_features2 = tf.convert_to_tensor(scipy.stats.norm.pdf(x3, mu3, sigma), tf.float32)
        self.cc_features3 = tf.convert_to_tensor(scipy.stats.norm.pdf(x4, mu4, sigma), tf.float32)

    def call(self, inputs, training=False):
        """Call logic.
        Args:
            1. encoder_hidden_states, Tensor (float32) shape [batch_size, length, hidden_size]
            2. durations_gt, Tensor (float32/int32) shape [batch_size, length]
        """
        encoder_hidden_states, durations_gt = inputs
        outputs, encoder_masks = self._length_regulator(
            encoder_hidden_states, durations_gt
        )
        return outputs, encoder_masks

    def _length_regulator(self, encoder_hidden_states, durations_gt):
        """Length regulator logic."""
        sum_durations = tf.reduce_sum(durations_gt, axis=-1)  # [batch_size]
        max_durations = tf.reduce_max(sum_durations)

        input_shape = tf.shape(encoder_hidden_states)
        batch_size = input_shape[0]
        hidden_size = input_shape[-1]

        # initialize output hidden states and encoder masking.
        # TODO add tflite_infer for coarse_coding
        if self.enable_tflite_convertible:
            # There is only 1 batch in inference, so we don't have to use
            # `tf.While` op with 3-D output tensor.
            repeats = durations_gt[0]
            real_length = tf.reduce_sum(repeats)
            pad_size = max_durations - real_length
            # masks : [max_durations]
            masks = tf.sequence_mask([real_length], max_durations, dtype=tf.int32)
            repeat_encoder_hidden_states = tf.repeat(
                encoder_hidden_states[0], repeats=repeats, axis=0
            )
            repeat_encoder_hidden_states = tf.expand_dims(
                tf.pad(repeat_encoder_hidden_states, [[0, pad_size], [0, 0]]), 0
            )  # [1, max_durations, hidden_size]

            outputs = repeat_encoder_hidden_states
            encoder_masks = masks
        else:
            outputs = tf.zeros(shape=[0, max_durations, hidden_size + self.addfeatures_num], dtype=encoder_hidden_states.dtype)
            # outputs = tf.zeros(shape=[0, max_durations, hidden_size], dtype=encoder_hidden_states.dtype)
            encoder_masks = tf.zeros(shape=[0, max_durations], dtype=tf.int32)

            def condition(
                i,
                batch_size,
                outputs,
                encoder_masks,
                encoder_hidden_states,
                durations_gt,
                max_durations,
            ):
                return tf.less(i, batch_size)

            def body(
                i,
                batch_size,
                outputs,
                encoder_masks,
                encoder_hidden_states,
                durations_gt,
                max_durations,
            ):

############################### ori ##################################
                # repeats = durations_gt[i]
                # real_length = tf.reduce_sum(repeats)
                # pad_size = max_durations - real_length
                # masks = tf.sequence_mask([real_length], max_durations, dtype=tf.int32)
                # repeat_encoder_hidden_states = tf.repeat(
                #     encoder_hidden_states[i], repeats=repeats, axis=0
                # )
                # repeat_encoder_hidden_states = tf.expand_dims(
                #     tf.pad(repeat_encoder_hidden_states, [[0, pad_size], [0, 0]]), 0
                # )  # [1, max_durations, hidden_size]
                # outputs = tf.concat([outputs, repeat_encoder_hidden_states], axis=0)
                # encoder_masks = tf.concat([encoder_masks, masks], axis=0)

############################### add duration info ##################################
                repeats = durations_gt[i]
                real_length = tf.reduce_sum(repeats)
                pad_size = max_durations - real_length
                masks = tf.sequence_mask([real_length], max_durations, dtype=tf.int32)
                repeat_encoder_hidden_states = tf.repeat(
                    encoder_hidden_states[i], repeats=repeats, axis=0
                )

                if self.addfeatures_num > 0:
                    # duration sum per phone
                    durdur = tf.repeat(repeats[:, tf.newaxis], repeats=repeats, axis=0)
                    durdur = tf.cast(durdur, encoder_hidden_states.dtype)

                    # acc duration
                    maskbool = tf.sequence_mask(repeats, tf.reduce_max(repeats))
                    durindex = tf.cumsum(tf.cast(maskbool, encoder_hidden_states.dtype), -1)
                    durindex = tf.boolean_mask(durindex, maskbool)[:, tf.newaxis]

                    # duration/(sum)
                    durindex = (durindex - 1) / durdur

                    # coarse_coding
                    indexs = tf.cast(durindex*100, tf.int32)
                    cc0 = tf.gather(self.cc_features0, 400+indexs)
                    cc1 = tf.gather(self.cc_features1, 300+indexs)
                    cc2 = tf.gather(self.cc_features2, 200+indexs)
                    cc3 = tf.gather(self.cc_features3, 100+indexs)
                    ccc = tf.concat([cc0, cc1, cc2, cc3], axis=-1)

                    if self.config.isaddur:
                        repeat_encoder_hidden_states = tf.concat([repeat_encoder_hidden_states, durdur], -1)

                    repeat_encoder_hidden_states = tf.concat([repeat_encoder_hidden_states, ccc], -1)

                repeat_encoder_hidden_states = tf.expand_dims(
                    tf.pad(repeat_encoder_hidden_states, [[0, pad_size], [0, 0]]), 0
                )  # [1, max_durations, hidden_size]
                outputs = tf.concat([outputs, repeat_encoder_hidden_states], axis=0)
                encoder_masks = tf.concat([encoder_masks, masks], axis=0)
                return [
                    i + 1,
                    batch_size,
                    outputs,
                    encoder_masks,
                    encoder_hidden_states,
                    durations_gt,
                    max_durations,
                ]

            # initialize iteration i.
            i = tf.constant(0, dtype=tf.int32)
            _, _, outputs, encoder_masks, _, _, _, = tf.while_loop(
                condition,
                body,
                [
                    i,
                    batch_size,
                    outputs,
                    encoder_masks,
                    encoder_hidden_states,
                    durations_gt,
                    max_durations,
                ],
                shape_invariants=[
                    i.get_shape(),
                    batch_size.get_shape(),
                    tf.TensorShape(
                        [
                            None,
                            None,
                            self.config.content_latent_dim,
                        ]
                    ),
                    tf.TensorShape([None, None]),
                    encoder_hidden_states.get_shape(),
                    durations_gt.get_shape(),
                    max_durations.get_shape(),
                ],
            )

        return outputs, encoder_masks