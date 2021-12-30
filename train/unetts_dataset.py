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
"""Dataset modules."""

import itertools
import logging
import os
import random

import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files

class UNETTSDurationDataset(AbstractDataset):
    """UNETTSDurationDataset"""

    def __init__(
        self,
        root_dir,
        charactor_query   = "*-ids.npy",
        duration_query    = "*-raw-durations.npy",
        stat_dur_query    = "*-stat-durations.npy",
        charactor_load_fn = np.load,
        duration_load_fn  = np.load,
        stat_dur_load_fn  = np.load,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            ...
        """
        # find all of charactor and mel files.
        charactor_files = sorted(find_files(root_dir, charactor_query))
        duration_files  = sorted(find_files(root_dir, duration_query))
        stat_dur_files  = sorted(find_files(root_dir, stat_dur_query))

        # assert the number of files
        assert len(duration_files) != 0, f"Not found any mels files in ${root_dir}."
        assert (
            len(charactor_files)
            == len(duration_files)
            == len(stat_dur_files)
        ), f"Number of charactor, duration and its stats files are different"

        if ".npy" in charactor_query:
            suffix = charactor_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in charactor_files]

        # set global params
        self.utt_ids           = utt_ids
        self.charactor_files   = charactor_files
        self.duration_files    = duration_files
        self.stat_dur_files    = stat_dur_files
        self.charactor_load_fn = charactor_load_fn
        self.duration_load_fn  = duration_load_fn
        self.stat_dur_load_fn  = stat_dur_load_fn

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids): 
            charactor_file = self.charactor_files[i]
            duration_file  = self.duration_files[i]
            stat_dur_file  = self.stat_dur_files[i]

            items = {
                "utt_ids"        : utt_id,
                "charactor_files": charactor_file,
                "duration_files" : duration_file,
                "stat_dur_files" : stat_dur_file,
            }

            yield items

    @tf.function
    def _load_data(self, items):
        charactor = tf.numpy_function(np.load, [items["charactor_files"]], tf.int32)
        duration  = tf.numpy_function(np.load, [items["duration_files"]], tf.float32)
        durstats  = tf.numpy_function(np.load, [items["stat_dur_files"]], tf.float32)

        items = {
            "utt_ids"      : items["utt_ids"],
            "char_ids"     : charactor,
            "char_lengths" : len(charactor),
            "duration_gts" : duration,
            "duration_stat": durstats
        }

        return items

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )

        # load data
        datasets = datasets.map(
            lambda items: self._load_data(items), tf.data.experimental.AUTOTUNE
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padded shapes
        padded_shapes = {
            "utt_ids"      : [],
            "char_ids"     : [None],
            "char_lengths" : [],
            "duration_gts" : [None],
            "duration_stat": [None],
        }

        datasets = datasets.padded_batch(batch_size, padded_shapes=padded_shapes)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {
            "utt_ids"        : tf.string,
            "charactor_files": tf.string,
            "duration_files" : tf.string,
            "stat_dur_files" : tf.string,
        }
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "UNETTSDurationDataset"

class UNETTSAcousDataset(AbstractDataset):
    """UNETTSAcousDataset"""

    def __init__(
        self,
        root_dir,
        charactor_query      = "*-ids.npy",
        duration_query       = "*-raw-durations.npy",
        mel_query            = "*-raw-mels.npy",
        embed_query          = "*-embeds.npy",
        mel_load_fn          = np.load,
        charactor_load_fn    = np.load,
        duration_load_fn     = np.load,
        embed_load_fn        = np.load,
        mel_length_threshold = 0,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            ...
        """
        # find all of charactor and mel files.
        charactor_files = sorted(find_files(root_dir, charactor_query))
        duration_files  = sorted(find_files(root_dir, duration_query))
        mel_files       = sorted(find_files(root_dir, mel_query))
        embed_files     = sorted(find_files(root_dir, embed_query))

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mels files in ${root_dir}."
        assert (
            len(charactor_files)
            == len(duration_files)
            == len(mel_files)
            == len(embed_files)
        ), f"Number of charactor, duration and its stats files are different"

        if ".npy" in charactor_query:
            suffix = charactor_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in charactor_files]

        # set global params
        self.utt_ids              = utt_ids
        self.charactor_files      = charactor_files
        self.duration_files       = duration_files
        self.mel_files            = mel_files
        self.embed_files          = embed_files
        self.charactor_load_fn    = charactor_load_fn
        self.duration_load_fn     = duration_load_fn
        self.mel_load_fn          = mel_load_fn
        self.embed_load_fn        = embed_load_fn
        self.mel_length_threshold = mel_length_threshold

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            charactor_file = self.charactor_files[i]
            duration_file  = self.duration_files[i]
            mel_file       = self.mel_files[i]
            embed_file     = self.embed_files[i]

            items = {
                "utt_ids"        : utt_id,
                "charactor_files": charactor_file,
                "duration_files" : duration_file,
                "mel_files"      : mel_file,
                "embed_files"    : embed_file,
            }

            yield items

    @tf.function
    def _load_data(self, items):
        charactor = tf.numpy_function(np.load, [items["charactor_files"]], tf.int32)
        duration  = tf.numpy_function(np.load, [items["duration_files"]], tf.int32)
        mel       = tf.numpy_function(np.load, [items["mel_files"]], tf.float32)
        embed     = tf.numpy_function(np.load, [items["embed_files"]], tf.float32)

        items = {
            "utt_ids"     : items["utt_ids"],
            "char_ids"    : charactor,
            "char_lengths": len(charactor),
            "mel_gts"     : mel,
            "mel_lengths" : len(mel),
            "duration_gts": duration,
            "embed"       : embed,
        }

        return items

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )

        # load data
        datasets = datasets.map(
            lambda items: self._load_data(items), tf.data.experimental.AUTOTUNE
        )

        # datasets = datasets.filter(
        #     lambda x: x["spe_lengths"] > self.mel_length_threshold
        # )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padded shapes
        padded_shapes = {
            "utt_ids"     : [],
            "char_ids"    : [None],
            "char_lengths": [],
            "mel_gts"     : [None, None],
            "mel_lengths" : [],
            "duration_gts": [None],
            "embed"       : [None],
        }

        datasets = datasets.padded_batch(batch_size, padded_shapes=padded_shapes)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {
            "utt_ids"        : tf.string,
            "charactor_files": tf.string,
            "mel_files"      : tf.string,
            "duration_files" : tf.string,
            "embed_files"    : tf.string,
        }
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "UNETTSAcousDataset"