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

import tensorflow as tf
from tensorflow_tts.models.moduls.core import *
from tensorflow_tts.models.moduls.conditional import ConditionalNormalization

class TFFastSpeechConditionalSelfOutput(tf.keras.layers.Layer):
    """Fastspeech output of self attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        self.normlayer = ConditionalNormalization(config)

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        '''
        hidden_states: [B, T, C]
        input_tensor:  [B, T, C]
        conds:         [B, 1, T]
        attention_mask:[B, T]
        '''
        hidden_states, input_tensor, conds, attention_mask = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.normlayer(hidden_states + input_tensor, conds, attention_mask)

        return hidden_states

class TFFastSpeechConditionalAttention(tf.keras.layers.Layer):
    """Fastspeech attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.self_attention = TFFastSpeechSelfAttention(config, name="self")
        self.dense_output   = TFFastSpeechConditionalSelfOutput(config, name="output")

    def call(self, inputs, training=False):
        '''
        input_tensor:   [B, T, C]
        conds:          [B, 1, C']
        attention_mask: [B, T]
        '''
        input_tensor, conds, attention_mask = inputs

        self_outputs = self.self_attention(
            [input_tensor, attention_mask], training=training
        )
        attention_output = self.dense_output(
            [self_outputs[0], input_tensor, conds, attention_mask], training=training
        )
        masked_attention_output = attention_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=attention_output.dtype
        )
        outputs = (masked_attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs

class TFFastSpeechConditionalOutput(tf.keras.layers.Layer):
    """Output module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.normlayer = ConditionalNormalization(config)
        self.dropout   = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        '''
        hidden_states: [B, T, C]
        input_tensor:  [B, T, C]
        conds:         [B, 1, T]
        attention_mask:[B, T]
        '''
        hidden_states, input_tensor, conds, attention_mask = inputs

        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.normlayer(hidden_states + input_tensor, conds, attention_mask)
        return hidden_states


class TFFastSpeechConditionalLayer(tf.keras.layers.Layer):
    """Fastspeech module (FFT module on the paper)."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.attention    = TFFastSpeechConditionalAttention(config, name="attention")
        self.intermediate = TFFastSpeechIntermediate(config, name="intermediate")
        self.bert_output  = TFFastSpeechConditionalOutput(config, name="output")

    def call(self, inputs, training=False):
        '''
        hidden_states:  [B, T, C]
        conds:          [B, 1, C']
        attention_mask: [B, T]
        '''
        hidden_states, conds, attention_mask = inputs

        attention_outputs = self.attention(
            [hidden_states, conds, attention_mask], training=training
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(
            [attention_output, attention_mask], training=training
        )
        layer_output = self.bert_output(
            [intermediate_output, attention_output, conds, attention_mask], training=training
        )
        masked_layer_output = layer_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=layer_output.dtype
        )
        outputs = (masked_layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them
        return outputs

class TFFastSpeechConditionalEncoder(tf.keras.layers.Layer):
    """Fast Speech encoder module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [
            TFFastSpeechConditionalLayer(config, name="layer_._{}".format(i))
            for i in range(config.num_hidden_layers)
        ]

    def call(self, inputs, training=False):
        '''
        hidden_states:  [B, T, C]
        conds:          [B, 1, C']
        attention_mask: [B, T]
        '''
        hidden_states, conds, attention_mask = inputs

        all_hidden_states = ()
        all_attentions = ()
        for _, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                [hidden_states, conds, attention_mask], training=training
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


class TFFastSpeechConditionalDecoder(TFFastSpeechConditionalEncoder):
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
        '''
        hidden_states: [B, T, C]
        conds:         [B, 1, C']
        encoder_mask:  [B, T]
        '''
        hidden_states, conds, encoder_mask = inputs

        if self.is_compatible_encoder is False:
            hidden_states = self.project_compatible_decoder(hidden_states)

        return super().call([hidden_states, conds, encoder_mask], training=training)