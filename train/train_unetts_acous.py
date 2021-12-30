# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
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
"""Train Unet-TTS"""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

import sys

# sys.path.append("../..")

import argparse
import logging
import os
import traceback

import numpy as np
import yaml

import tensorflow_tts
from train.unetts_dataset import UNETTSAcousDataset
from tensorflow_tts.configs import UNETTSAcousConfig
from tensorflow_tts.models import TFUNETTSContentPretrain, TFUNETTSAcous
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.utils import calculate_loss_norm_lens, return_strategy
from tensorflow_tts.audio_process.audio_spec import AudioMelSpec


class UNETTSAcousTrainer(Seq2SeqBasedTrainer):
    """UNETTSAcousTrainer."""

    def __init__(
        self, config, strategy, steps=0, epochs=0, is_mixed_precision=False,
    ):
        """Initialize trainer.
        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_mixed_precision (bool): Use mixed precision or not.
        """
        super(UNETTSAcousTrainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "mel_before",
            "content_loss",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.feature_handle = AudioMelSpec(**config["feat_params"])

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        # self.bce = tf.keras.losses.BinaryCrossentropy(
        #     reduction=tf.keras.losses.Reduction.NONE
        # )

    def compute_per_example_losses(self, batch, outputs):
        """Compute per example losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        mel_before, content_latents, content_latent_pred = outputs

        # mse, 0.01; mae, 0.05
        mel_loss_before = calculate_loss_norm_lens(batch["mel_gts"], mel_before,   self.mae, batch["mel_lengths"])
        content_loss    = calculate_loss_norm_lens(content_latents, content_latent_pred, self.mse, batch["mel_lengths"])

        per_example_losses = (
            mel_loss_before + content_loss
        )

        dict_metrics_losses = {
            "mel_before": mel_loss_before,
            "content_loss" : content_loss,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # predict with tf.function.
        mel_before, *_ = self.one_step_predict(batch)

        mel_gts = batch["mel_gts"].numpy()
        frame_real_length = batch["mel_lengths"].numpy()

        # convert to tensor.
        # here we just take a sample at first replica.
        try: 
            mel_before  = mel_before.values[0].numpy()
        except Exception: 
            mel_before  = mel_before.numpy()

        # check directory
        utt_ids = batch["utt_ids"].numpy()
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for i, utt in enumerate(utt_ids):
            figname = os.path.join(dirname, f"{utt}.png")
            wavname = os.path.join(dirname, f"{utt}.wav")

            self.feature_handle.compare_plot(mel_gts[i],
                                             mel_before[i],
                                             filepath=figname,
                                             frame_real_len=frame_real_length[i],
                                             text=None)

            if self.epochs > self.config["wav_output_epochs"] and i < self.config["results_num"]:
                audio = self.feature_handle.inv_mel_spectrogram(mel_before[i][:frame_real_length[i]])
                self.feature_handle.save_wav(audio, wavname)


class UNETTSContentPreTrainer(Seq2SeqBasedTrainer):
    """UNETTSContentPreTrainer"""

    def __init__(
        self, config, strategy, steps=0, epochs=0, is_mixed_precision=False,
    ):
        """Initialize trainer.
        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_mixed_precision (bool): Use mixed precision or not.
        """
        super(UNETTSContentPreTrainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "mel_before",
            "mel_after",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.feature_handle = AudioMelSpec(**config["feat_params"])

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        # self.bce = tf.keras.losses.BinaryCrossentropy(
        #     reduction=tf.keras.losses.Reduction.NONE
        # )

    def compute_per_example_losses(self, batch, outputs):
        """Compute per example losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        mel_before, mel_after = outputs

        # mse, 0.01; mae, 0.05
        mel_loss_before = calculate_loss_norm_lens(batch["mel_gts"], mel_before, self.mae, batch["mel_lengths"])
        mel_loss_after  = calculate_loss_norm_lens(batch["mel_gts"], mel_after,  self.mae, batch["mel_lengths"])

        per_example_losses = (
            mel_loss_before + mel_loss_after
        )

        dict_metrics_losses = {
            "mel_before": mel_loss_before,
            "mel_after" : mel_loss_after,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # predict with tf.function.
        outputs = self.one_step_predict(batch)

        mel_before, mel_after = outputs

        mel_gts = batch["mel_gts"].numpy()
        frame_real_length = batch["mel_lengths"].numpy()

        # convert to tensor.
        # here we just take a sample at first replica.
        try: 
            mel_before  = mel_before.values[0].numpy()
            mel_after   = mel_after.values[0].numpy()
        except Exception: 
            mel_before  = mel_before.numpy()
            mel_after   = mel_after.numpy()

        # check directory
        utt_ids = batch["utt_ids"].numpy()
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for i, utt in enumerate(utt_ids):
            figname = os.path.join(dirname, f"{utt}.png")
            wavname = os.path.join(dirname, f"{utt}.wav")

            self.feature_handle.compare_plot(mel_gts[i],
                                             mel_after[i],
                                             filepath=figname,
                                             frame_real_len=frame_real_length[i],
                                             text=None)

            if self.epochs > self.config["wav_output_epochs"] and i < self.config["results_num"]:
                audio = self.feature_handle.inv_mel_spectrogram(mel_after[i][:frame_real_length[i]])
                self.feature_handle.save_wav(audio, wavname)

def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train model (See detail in tensorflow_tts/models)"
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        type=str,
        help="directory including training data. ",
    )
    parser.add_argument(
        "--dev-dir",
        default=None,
        type=str,
        help="directory including development data. ",
    )
    parser.add_argument(
        "--content_training", default=0, type=int, help="is need to training context_model"
    )
    parser.add_argument(
        "--content_pretrained_path", type=str, required=True, help="directory to content_pretrained_path"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--data_config", type=str, required=True, help="yaml format data_process file."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--mixed_precision",
        default=0,
        type=int,
        help="using mixed precision for generator or not.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        nargs="?",
        help='pretrained weights .h5 file to load weights from. Auto-skips non-matching layers',
    )
    

    args = parser.parse_args()

    # return strategy
    STRATEGY = return_strategy()

    # set mixed precision config
    if args.mixed_precision == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    args.mixed_precision = bool(args.mixed_precision)
    # args.use_norm = bool(args.use_norm)

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # check arguments
    if args.train_dir is None:
        raise ValueError("Please specify --train-dir")
    if args.dev_dir is None:
        raise ValueError("Please specify --valid-dir")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    with open(args.data_config) as f:
        data_config = yaml.load(f, Loader=yaml.Loader)
    config.update(data_config)

    config.update(vars(args))
    config["version"] = tensorflow_tts.__version__
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    config["content_training"] = True if config["content_training"] else False

    if config["content_training"]:
        print("*"*50)
        print("*"*20 + "Training Content Encoder ......" + "*"*20)
        print("*"*50)
        assert config["unetts_acous_context_pre_params"]["num_mels"] == config["feat_params"]["num_mels"]
    else:
        assert config["unetts_acous_params"]["num_mels"] == config["feat_params"]["num_mels"]

    # get dataset
    # if config["remove_short_samples"]:
    #     mel_length_threshold = config["mel_length_threshold"]
    # else:
    #     mel_length_threshold = None

    # if config["format"] == "npy":
    #     charactor_query = "*-ids.npy"
    #     mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
    #     duration_query = "*-durations.npy"
    #     lf0_query = "*-raw-lf0.npy"
    #     energy_query = "*-raw-energy.npy"
    # else:
    #     raise ValueError("Only npy are supported.")

    # define train/valid dataset
    train_dataset = UNETTSAcousDataset(
        root_dir=args.train_dir,
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
    )

    valid_dataset = UNETTSAcousDataset(
        root_dir=args.dev_dir,
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
    )

    # define trainer
    if config["content_training"]:
        trainer = UNETTSContentPreTrainer(
            config=config,
            strategy=STRATEGY,
            steps=0,
            epochs=0,
            is_mixed_precision=args.mixed_precision,
        )
    else:
        trainer = UNETTSAcousTrainer(
            config=config,
            strategy=STRATEGY,
            steps=0,
            epochs=0,
            is_mixed_precision=args.mixed_precision,
        )

    with STRATEGY.scope():
        # define model
        if config["content_training"]:
            model = TFUNETTSContentPretrain(
                config=UNETTSAcousConfig(**config["unetts_acous_context_pre_params"])
            )
        else:
            model = TFUNETTSAcous(
                config=UNETTSAcousConfig(**config["unetts_acous_params"])
            )

        model._build()

        if not config["content_training"]:
            logging.info("Model content_pretrained_path: {}".format(config["content_pretrained_path"]))
            try:
                # TODO
                model.text_encoder_weight_load(config["content_pretrained_path"])
                model.freezen_encoder()
            except:
                logging.error("Error: Model embedding and text_encoder")
                exit(1)

        model.summary()

        if len(args.pretrained) > 1:
            model.load_weights(args.pretrained, by_name=True, skip_mismatch=True)
            logging.info(f"Successfully loaded pretrained weight from {args.pretrained}.")

        # AdamW for model
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
            decay_steps=config["optimizer_params"]["decay_steps"],
            end_learning_rate=config["optimizer_params"]["end_learning_rate"],
        )

        learning_rate_fn = WarmUp(
            initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=int(
                config["train_max_steps"]
                * config["optimizer_params"]["warmup_proportion"]
            ),
        )

        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=config["optimizer_params"]["weight_decay"],
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )

        _ = optimizer.iterations

    # compile trainer
    trainer.compile(model=model, optimizer=optimizer)

    # start training
    try:
        trainer.fit(
            train_dataset,
            valid_dataset,
            saved_path=os.path.join(config["outdir"], "checkpoints/"),
            resume=args.resume,
        )
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
    except:
        error = traceback.format_exc()
        print(error)
        with open(os.path.join(config["outdir"], "error.txt"), 'a') as fw:
            fw.write(error + "\n")


if __name__ == "__main__":
    main()
