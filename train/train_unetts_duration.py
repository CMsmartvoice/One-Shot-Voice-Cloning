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
"""Train MultiSPKCLDuration."""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

import sys

sys.path.append("../..")

import argparse
import logging
import os
import traceback

import numpy as np
import yaml
import matplotlib.pyplot as plt

import tensorflow_tts
from train.unetts_dataset import UNETTSDurationDataset
from tensorflow_tts.configs import UNETTSDurationConfig
from tensorflow_tts.models import TFUNETTSDuration
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.utils import (calculate_loss_norm_lens, return_strategy)


class UNETTSDurationTrainer(Seq2SeqBasedTrainer):
    """UNETTSDurationTrainer"""

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
        super(UNETTSDurationTrainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "dur_loss",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.huber = tf.keras.losses.Huber(
            delta = 2.0,
            reduction=tf.keras.losses.Reduction.NONE
        )

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

        # log_duration = tf.math.log(
        #     tf.cast(tf.math.add(batch["duration_gts"], 1), tf.float32)
        # )

        per_example_losses = calculate_loss_norm_lens(batch["duration_gts"], outputs,  self.mae, batch["char_lengths"])

        dict_metrics_losses = {
            "dur_loss": per_example_losses,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        outputs = self.one_step_predict(batch)

        try:
            dur_preds = outputs.values[0].numpy()
        except Exception:
            dur_preds = outputs.numpy()

        dur_gts      = batch["duration_gts"].numpy()
        char_lengths = batch["char_lengths"].numpy()
        utt_ids      = batch["utt_ids"].numpy()

        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for i, utt_id in enumerate(utt_ids):
            figname = os.path.join(dirname, f"{utt_id}.png")
            plt.figure(figsize=(10, 4))
            plt.plot(dur_gts[i][:char_lengths[i]],   'b--o', markersize=6)
            plt.plot(dur_preds[i][:char_lengths[i]], 'r-x', markersize=10)
            plt.grid()
            plt.legend(("gst", "pred"))
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()


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
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
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
    config.update(vars(args))
    config["version"] = tensorflow_tts.__version__
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # # get dataset
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
    train_dataset = UNETTSDurationDataset(
        root_dir=args.train_dir
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
    )

    valid_dataset = UNETTSDurationDataset(
        root_dir=args.dev_dir
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
    )

    # define trainer
    trainer = UNETTSDurationTrainer(
        config=config,
        strategy=STRATEGY,
        steps=0,
        epochs=0,
        is_mixed_precision=args.mixed_precision,
    )

    with STRATEGY.scope():
        # define model
        model = TFUNETTSDuration(
            config=UNETTSDurationConfig(**config["unetts_duration"])
        )
        model._build()
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
