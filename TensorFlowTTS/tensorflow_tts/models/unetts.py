import tensorflow as tf
import numpy as np

from tensorflow_tts.models.moduls.core import (
    TFFastSpeechEmbeddings,
    TFFastSpeechEncoder,
    TFFastSpeechDecoder,
    TFTacotronPostnet,
    TFFastSpeechLengthRegulator,
    TFFastSpeechVariantPredictor,
    TFFastSpeechDurationPredictor
)
from tensorflow_tts.models.moduls.core2 import TFFastSpeechConditionalDecoder
from tensorflow_tts.models.moduls.adain_en_de_code import (
    AadINEncoder, AdaINDecoder
)

'''
###############################################################################
#############################  Duration #######################################
###############################################################################
'''

class TFUNETTSDuration(tf.keras.Model):
    def __init__(self, config, **kwargs):
        """Init layers for UNETTSDuration."""
        self.enable_tflite_convertible = kwargs.pop("enable_tflite_convertible", False)
        super().__init__(**kwargs)

        self.embeddings = TFFastSpeechEmbeddings(config, name="embeddings")

        self.encoder = TFFastSpeechEncoder(
            config.encoder_self_attention_params, name = "encoder"
        )

        self.duration_predictor = TFFastSpeechDurationPredictor(
            config, name = "duration_predictor"
        )

        self.duration_stat_cal = tf.keras.layers.Dense(4, use_bias=False,
                                                    kernel_initializer=tf.constant_initializer(
                                                        [[0.97, 0.01, 0.01, 0.01],
                                                        [0.01, 0.97, 0.01, 0.01],
                                                        [0.01, 0.01, 0.97, 0.01],
                                                        [0.01, 0.01, 0.01, 0.97]]
                                                    ),
                                                    kernel_constraint=tf.keras.constraints.NonNeg(),
                                                    name="duration_stat_cal")

        self.setup_inference_fn()

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        char_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        duration_stat = tf.convert_to_tensor([[1., 1., 1., 1.]], tf.float32)
        self(char_ids, duration_stat)

    def call(
        self, char_ids, duration_stat, training=False, **kwargs,
    ):
        """Call logic."""
        attention_mask = tf.math.not_equal(char_ids, 0)
        sheng_mask = char_ids < 27
        yun_mask   = char_ids > 26

        duration_stat = self.duration_stat_cal(duration_stat)

        sheng_mean, sheng_std, yun_mean, yun_std = \
            duration_stat[:,0][:, None], duration_stat[:,1][:, None], duration_stat[:,2][:, None], duration_stat[:,3][:, None]

        embedding_output = self.embeddings(char_ids)

        encoder_output             = self.encoder([embedding_output, attention_mask], training=training)
        last_encoder_hidden_states = encoder_output[0]

        duration_outputs = self.duration_predictor([last_encoder_hidden_states, attention_mask])

        sheng_outputs = duration_outputs * sheng_std + sheng_mean
        sheng_outputs = sheng_outputs * tf.cast(sheng_mask, tf.float32)

        yun_outputs = duration_outputs * yun_std + yun_mean
        yun_outputs = yun_outputs * tf.cast(yun_mask, tf.float32)

        duration_outputs = sheng_outputs + yun_outputs
        duration_outputs = tf.nn.relu(duration_outputs * tf.cast(attention_mask, tf.float32))

        return duration_outputs

    def _inference(self, char_ids, duration_stat, **kwargs):
        """Call logic."""
        attention_mask = tf.math.not_equal(char_ids, 0)
        sheng_mask = char_ids < 27
        yun_mask   = char_ids > 26

        duration_stat = self.duration_stat_cal(duration_stat)

        sheng_mean, sheng_std, yun_mean, yun_std = \
            duration_stat[:,0][:, None], duration_stat[:,1][:, None], duration_stat[:,2][:, None], duration_stat[:,3][:, None]

        embedding_output = self.embeddings(char_ids, training=False)

        encoder_output             = self.encoder([embedding_output, attention_mask], training=False)
        last_encoder_hidden_states = encoder_output[0]

        duration_outputs = self.duration_predictor([last_encoder_hidden_states, attention_mask])

        sheng_outputs = duration_outputs * sheng_std + sheng_mean
        sheng_outputs = sheng_outputs * tf.cast(sheng_mask, tf.float32)

        yun_outputs = duration_outputs * yun_std + yun_mean
        yun_outputs = yun_outputs * tf.cast(yun_mask, tf.float32)

        duration_outputs = sheng_outputs + yun_outputs
        duration_outputs = tf.nn.relu(duration_outputs * tf.cast(attention_mask, tf.float32))

        return duration_outputs

    def setup_inference_fn(self):
        self.inference = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="char_ids"),
                tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="duration_stat"),
            ],
        )

        self.inference_tflite = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[1, None], dtype=tf.int32, name="char_ids"),
                tf.TensorSpec(shape=[1, None], dtype=tf.float32, name="duration_stat"),
            ],
        )

'''
###############################################################################
################################ Acous ########################################
###############################################################################
'''

class ContentEncoder(tf.keras.Model):
    def __init__(self, config, **kwargs):
        """Init layers for ContentEncoder."""
        self.enable_tflite_convertible = kwargs.pop("enable_tflite_convertible", False)
        super().__init__(**kwargs)

        self.embeddings = TFFastSpeechEmbeddings(config, name="embeddings")

        self.encoder    = TFFastSpeechEncoder(
            config.encoder_self_attention_params, name="encoder"
        )

        self.length_regulator = TFFastSpeechLengthRegulator(
            config,
            enable_tflite_convertible = self.enable_tflite_convertible,
            name                      = "length_regulator",
        )

    def call(self, char_ids, duration_gts, training=False):
        attention_mask = tf.math.not_equal(char_ids, 0)

        embedding_output = self.embeddings(char_ids)

        encoder_output = self.encoder([embedding_output, attention_mask], training=training)
        last_encoder_hidden_states = encoder_output[0]

        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_gts], training=training
        )

        return length_regulator_outputs, encoder_masks

class TFUNETTSAcous(tf.keras.Model):
    """TF UNETTSAcous module."""

    def __init__(self, config, **kwargs):
        """Init layers for UNETTSAcous."""
        self.enable_tflite_convertible = kwargs.pop("enable_tflite_convertible", False)
        super().__init__(**kwargs)

        self.config = config

        self.content_encoder = ContentEncoder(
            config,
            enable_tflite_convertible=self.enable_tflite_convertible,
            name="content_encoder"
        )

        self.src_mel_encoder = AadINEncoder(config, name="src_mel_encoder")

        self.tar_mel_decoder = AdaINDecoder(config, name="tar_mel_decoder")

        self.setup_inference_fn()

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        char_ids     = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        duration_gts = tf.convert_to_tensor([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]], tf.int32)
        mel_src      = tf.random.normal([1, 30, self.config.num_mels], dtype=tf.float32)
        self(char_ids, duration_gts, mel_src)

    def text_encoder_weight_load(self, content_encoder_path):
        self.content_encoder.load_weights(content_encoder_path)

    def freezen_encoder(self):
        self.content_encoder.trainable = False

    def call(
        self, char_ids, duration_gts, mel_src, training=False, **kwargs,
    ):
        """Call logic."""
        content_latents, encoder_masks = self.content_encoder(char_ids, duration_gts, training=False)
        content_latents = content_latents * tf.cast(tf.expand_dims(encoder_masks, axis=2), content_latents.dtype)

        content_latent_pred, means, stds = self.src_mel_encoder(mel_src, encoder_masks)
        content_latent_pred = content_latent_pred * tf.cast(tf.expand_dims(encoder_masks, axis=2), content_latent_pred.dtype)

        mel_before = self.tar_mel_decoder(content_latents, (content_latent_pred, means, stds), encoder_masks)
        mel_before = mel_before * tf.cast(tf.expand_dims(encoder_masks, axis=2), mel_before.dtype)

        return (mel_before, content_latents, content_latent_pred)

    def _inference(self, char_ids, duration_gts, mel_src, **kwargs):
        """Call logic."""
        content_latents, encoder_masks = self.content_encoder(char_ids, duration_gts, training=False)

        tmp_masks = tf.ones([tf.shape(mel_src)[0], tf.shape(mel_src)[1]], dtype=tf.bool)
        _, means, stds = self.src_mel_encoder(mel_src, tmp_masks)

        mel_before = self.tar_mel_decoder(content_latents, (_, means, stds), encoder_masks)

        return mel_before, means, stds

    def extract_dur_pos_embed(self, mel_src):
        tmp_masks = tf.ones([tf.shape(mel_src)[0], tf.shape(mel_src)[1]], dtype=tf.bool)
        content_latent_pred, _, _ = self.src_mel_encoder(mel_src, tmp_masks)
        return content_latent_pred[:, :, -4:]

    def setup_inference_fn(self):
        self.inference = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="char_ids"),
                tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="duration_gts"),
                tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="mel_src"),
            ],
        )

        self.inference_tflite = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[1, None], dtype=tf.int32, name="char_ids"),
                tf.TensorSpec(shape=[1, None], dtype=tf.int32, name="duration_gts"),
                tf.TensorSpec(shape=[1, None, None], dtype=tf.float32, name="mel_src"),
            ],
        )

class TFUNETTSContentPretrain(tf.keras.Model):
    """UNETTSContentPretrain"""

    def __init__(self, config, **kwargs):
        self.enable_tflite_convertible = kwargs.pop("enable_tflite_convertible", False)
        super().__init__(**kwargs)

        self.config = config

        self.content_encoder = ContentEncoder(
            config,
            enable_tflite_convertible=self.enable_tflite_convertible,
            name="content_encoder"
        )

        if self.config.decoder_is_conditional:
            self.decoder = TFFastSpeechConditionalDecoder(
                config.decoder_self_attention_conditional_params,
                is_compatible_encoder = True,
                name                  = "decoder",
            )
        else:
            self.decoder = TFFastSpeechDecoder(
                config.decoder_self_attention_params,
                is_compatible_encoder = False,
                name                  = "decoder",
            )

        self.mel_dense = tf.keras.layers.Dense(units=config.num_mels, dtype=tf.float32, name="mel_before")
        self.postnet   = TFTacotronPostnet(config=config, dtype=tf.float32, name="postnet")

        self.setup_inference_fn()

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        char_ids     = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        duration_gts = tf.convert_to_tensor([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]], tf.int32)
        embed        = tf.random.normal([1, 256], dtype=tf.float32)
        self(char_ids, duration_gts, embed)

    def content_encoder_weight_save(self, path):
        self.content_encoder.save_weights(path)

    def call(
        self, char_ids, duration_gts, embed, training=False, **kwargs,
    ):
        """Call logic."""
        content_latents, encoder_masks = self.content_encoder(char_ids, duration_gts, training=training)

        if self.config.decoder_is_conditional:
            decoder_output = self.decoder(
                [content_latents, tf.expand_dims(embed, 1), encoder_masks],
                training=training,
            )
        else:
            # TODO
            frame_num = tf.reduce_max(tf.reduce_sum(duration_gts, 1))
            expand_embeds = tf.tile(tf.expand_dims(embed, 1), [1, frame_num, 1])
            content_latents = tf.concat([content_latents, expand_embeds], -1)

            decoder_output = self.decoder(
                [content_latents, encoder_masks],
                training=training,
            )

        last_decoder_hidden_states = decoder_output[0]

        spebap_before = self.mel_dense(last_decoder_hidden_states)
        spebap_before = spebap_before * tf.cast(tf.expand_dims(encoder_masks, axis=2), spebap_before.dtype)

        spebap_after = self.postnet([spebap_before, encoder_masks], training=training) + spebap_before

        outputs = (spebap_before, spebap_after)
        return outputs

    def _inference(self, char_ids, duration_gts, embed, **kwargs):
        """Call logic."""
        content_latents, encoder_masks = self.content_encoder(char_ids, duration_gts, training=False)

        if self.config.decoder_is_conditional:
            decoder_output = self.decoder(
                [content_latents, tf.expand_dims(embed, 1), encoder_masks],
                training=False,
            )
        else:
            # TODO
            frame_num = tf.reduce_max(tf.reduce_sum(duration_gts, 1))
            expand_embeds = tf.tile(tf.expand_dims(embed, 1), [1, frame_num, 1])
            content_latents = tf.concat([content_latents, expand_embeds], -1)

            decoder_output = self.decoder(
                [content_latents, encoder_masks],
                training=False,
            )

        last_decoder_hidden_states = decoder_output[0]

        spebap_before = self.mel_dense(last_decoder_hidden_states)
        spebap_before = spebap_before * tf.cast(tf.expand_dims(encoder_masks, axis=2), spebap_before.dtype)

        spebap_after = self.postnet([spebap_before, encoder_masks], training=False) + spebap_before

        outputs = (spebap_before, spebap_after)
        return outputs

    def setup_inference_fn(self):
        self.inference = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="char_ids"),
                tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="duration_gts"),
                tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="embed")
            ],
        )

        self.inference_tflite = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[1, None], dtype=tf.int32, name="char_ids"),
                tf.TensorSpec(shape=[1, None], dtype=tf.int32, name="duration_gts"),
                tf.TensorSpec(shape=[1, None], dtype=tf.float32, name="embed")
            ],
        )