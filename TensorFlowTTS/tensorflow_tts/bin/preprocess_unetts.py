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
"""Perform preprocessing, with raw feature extraction and normalization of train/valid split."""

import argparse
import logging
import os
import yaml

import numpy as np

from functools import partial
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tensorflow_tts.processor.multispk_voiceclone import MultiSPKVoiceCloneProcessor
from tensorflow_tts.processor.multispk_voiceclone import AISHELL_CHN_SYMBOLS

from tensorflow_tts.audio_process.audio_spec import AudioMelSpec

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import tensorflow as tf

SEED =  2021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

_feats_handle = None

def parse_and_config():
    """Parse arguments and set configuration parameters."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and text features "
        "(See detail in tensorflow_tts/bin/preprocess_dataset.py)."
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        required=True,
        help="Directory containing the dataset files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        required=True,
        help="Output directory where features will be saved.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multispk_voiceclone",
        choices=["multispk_voiceclone"],
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--during_train",
        type=int,
        default=0,
        choices=[0, 1],
        help="0-False, 1-True: trainging during model"
    )
    parser.add_argument(
        "--all_train",
        type=int,
        default=0,
        choices=[0, 1],
        help="0-False, 1-True: trainging f0 model"
    )
    parser.add_argument(
        "--mfaed_txt",
        type=str,
        default=None,
        required=True,
        help="mfa results txt"
    )
    parser.add_argument(
        "--wavs_dir",
        type=str,
        default=None,
        required=True,
        help="wav dir"
    )
    parser.add_argument(
        "--spkinfo_dir",
        type=str,
        default=None,
        required=True,
        help="spkinfo dir"
    )
    parser.add_argument(
        "--embed_dir",
        type=str,
        default=None,
        required=True,
        help="embed dir"
    )
    parser.add_argument(
        "--unseen_dir",
        type=str,
        default=None,
        required=True,
        help="unseen speaker dir"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="YAML format configuration file."
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=4,
        required=False,
        help="Number of CPUs to use in parallel.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        required=False,
        help="Proportion of files to use as test dataset.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging level. 0: DEBUG, 1: INFO and WARNING, 2: INFO, WARNING, and ERROR",
    )
    args = parser.parse_args()

    # set logger
    FORMAT = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    log_level = {0: logging.DEBUG, 1: logging.WARNING, 2: logging.ERROR}
    logging.basicConfig(level=log_level[args.verbose], format=FORMAT)

    # load config
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    config.update(vars(args))
    # config checks
    assert config["format"] == "npy", "'npy' is the only supported format."
    return config

'''
###############################################################################
#############################  Duration #######################################
###############################################################################
'''

def preprocess_duration():
    """Run preprocessing process and compute statistics for normalizing."""
    config = parse_and_config()

    dataset_processor = {
        "multispk_voiceclone": MultiSPKVoiceCloneProcessor,
    }

    dataset_symbol = {
        "multispk_voiceclone": AISHELL_CHN_SYMBOLS,
    }

    dataset_cleaner = {
        "multispk_voiceclone": None,
    }

    logging.info(f"Selected '{config['dataset']}' processor.")
    processor = dataset_processor[config["dataset"]](
        config["rootdir"],
        symbols       = dataset_symbol[config["dataset"]],
        cleaner_names = dataset_cleaner[config["dataset"]],
        during_train  = True if config["during_train"] else False,
        mfaed_txt     = config["mfaed_txt"],
        wavs_dir      = config["wavs_dir"],
        embed_dir     = config["embed_dir"],
        spkinfo_dir   = config["spkinfo_dir"],
        unseen_dir    = config["unseen_dir"]
    )

    # check output directories
    build_dir = lambda x: [
        os.makedirs(os.path.join(config["outdir"], x, y), exist_ok=True)
        for y in ["ids", "raw-durations", "stat-durations"]
    ]
    build_dir("train")
    build_dir("valid")

    # save pretrained-processor to feature dir
    processor._save_mapper(
        os.path.join(config["outdir"], f"{config['dataset']}_mapper.json"),
        extra_attrs_to_save={"pinyin_dict": processor.pinyin_dict}
        if config["dataset"] == "multispk_voiceclone" else {},
    )

    # build train test split
    _Y = [i[0] for i in processor.items]
    train_split, valid_split = train_test_split(
        processor.items,
        test_size=config["test_size"],
        random_state=42,
        shuffle=True,
        stratify=_Y
    )
    logging.info(f"Training items: {len(train_split)}")
    logging.info(f"Validation items: {len(valid_split)}")

    train_utt_ids = [x[1] for x in train_split]
    valid_utt_ids = [x[1] for x in valid_split]

    # save train and valid utt_ids to track later
    np.save(os.path.join(config["outdir"], "train_utt_ids.npy"), train_utt_ids, allow_pickle=False)
    np.save(os.path.join(config["outdir"], "valid_utt_ids.npy"), valid_utt_ids, allow_pickle=False)

    config["none_pinyin_symnum"] = processor.none_pinyin_symnum

    # define map iterator
    def iterator_data(items_list):
        for item in items_list:
            yield processor.get_one_sample(item)

    train_iterator_data = iterator_data(train_split)
    valid_iterator_data = iterator_data(valid_split)

    p = Pool(config["n_cpus"])

    # preprocess train files and get statistics for normalizing
    partial_fn = partial(gen_duration_features, config=config)
    train_map = p.imap(
        partial_fn,
        tqdm(train_iterator_data, total=len(train_split), desc="[Preprocessing train]"),
        chunksize=10,
    )

    for item in train_map:
        save_duration_to_file(item, "train", config)

    # preprocess valid files
    partial_fn = partial(gen_duration_features, config=config)
    valid_map = p.imap(
        partial_fn,
        tqdm(valid_iterator_data, total=len(valid_split), desc="[Preprocessing valid]"),
        chunksize=10,
    )
    for item in valid_map:
        save_duration_to_file(item, "valid", config)

    """
        sample = {
            "speaker_name": spkname,
            "filename"    : filename,
            "wav_path"    : wav_path,
            "text_ids"    : text_ids,
            "durs"        : durs,
            "embed_path"  : embed_path,
            "rate"        : self.target_rate,
        }
    """
def gen_duration_features(item, config):
    text_ids = item["text_ids"]
    durs = item["durs"]
    assert len(text_ids) == len(durs)
    none_phnum = config["none_pinyin_symnum"]

    shengmu = []
    yunmu = []
    is_shengmu = True
    for t_id, dur in zip(text_ids, durs):
        if t_id < none_phnum:
            continue

        if is_shengmu:
            shengmu.append(dur)
            is_shengmu = False
        else:
            yunmu.append(dur)
            is_shengmu = True

    assert len(shengmu) == len(yunmu)

    dur_stats = np.array([np.mean(shengmu), np.std(shengmu), np.mean(yunmu), np.std(yunmu)])

    item["text_ids"]  = np.array(text_ids)
    item["durs"]      = np.array(durs)
    item["dur_stats"] = dur_stats

    return item

def save_duration_to_file(features, subdir, config):
    filename = features["filename"]

    if config["format"] == "npy":
        save_list = [
            (features["text_ids"],  "ids",              "ids",              np.int32),
            (features["durs"],      "raw-durations",    "raw-durations",    np.float32),
            (features["dur_stats"], "stat-durations",   "stat-durations",   np.float32),
        ]
        for item, name_dir, name_file, fmt in save_list:
            np.save(
                os.path.join(
                    config["outdir"], subdir, name_dir, f"{filename}-{name_file}.npy"
                ),
                item.astype(fmt),
                allow_pickle=False,
            )
    else:
        raise ValueError("'npy' is the only supported format.")


'''
###############################################################################
################################ Acous ########################################
###############################################################################
'''
def preprocess_acous():
    """Run preprocessing process and compute statistics for normalizing."""
    config = parse_and_config()

    dataset_processor = {
        "multispk_voiceclone": MultiSPKVoiceCloneProcessor,
    }

    dataset_symbol = {
        "multispk_voiceclone": AISHELL_CHN_SYMBOLS,
    }

    dataset_cleaner = {
        "multispk_voiceclone": None,
    }

    logging.info(f"Selected '{config['dataset']}' processor.")
    processor = dataset_processor[config["dataset"]](
        config["rootdir"],
        symbols       = dataset_symbol[config["dataset"]],
        cleaner_names = dataset_cleaner[config["dataset"]],
        all_train     = True if config["all_train"] else False,
        mfaed_txt     = config["mfaed_txt"],
        wavs_dir      = config["wavs_dir"],
        embed_dir     = config["embed_dir"],
        spkinfo_dir   = config["spkinfo_dir"],
        unseen_dir    = config["unseen_dir"]
    )

    # check output directories
    build_dir = lambda x: [
        os.makedirs(os.path.join(config["outdir"], x, y), exist_ok=True)
        for y in ["ids", "raw-durations", 
                  "raw-mels", "embeds"]
    ]
    build_dir("train")
    build_dir("valid")

    # save pretrained-processor to feature dir
    processor._save_mapper(
        os.path.join(config["outdir"], f"{config['dataset']}_mapper.json"),
        extra_attrs_to_save={"pinyin_dict": processor.pinyin_dict}
        if config["dataset"] == "multispk_voiceclone" else {},
    )

    # build train test split
    _Y = [i[0] for i in processor.items]
    train_split, valid_split = train_test_split(
        processor.items,
        test_size=config["test_size"],
        random_state=42,
        shuffle=True,
        stratify=_Y
    )
    logging.info(f"Training items: {len(train_split)}")
    logging.info(f"Validation items: {len(valid_split)}")

    train_utt_ids = [x[1] for x in train_split]
    valid_utt_ids = [x[1] for x in valid_split]

    # save train and valid utt_ids to track later
    np.save(os.path.join(config["outdir"], "train_utt_ids.npy"), train_utt_ids, allow_pickle=False)
    np.save(os.path.join(config["outdir"], "valid_utt_ids.npy"), valid_utt_ids, allow_pickle=False)

    # config["none_pinyin_symnum"] = processor.none_pinyin_symnum

    # define map iterator
    def iterator_data(items_list):
        for item in items_list:
            yield processor.get_one_sample(item)

    train_iterator_data = iterator_data(train_split)
    valid_iterator_data = iterator_data(valid_split)

    p = Pool(config["n_cpus"])

    # preprocess train files and get statistics for normalizing
    partial_fn = partial(gen_acous_features, config=config)
    train_map = p.imap_unordered(
        partial_fn,
        tqdm(train_iterator_data, total=len(train_split), desc="[Preprocessing train]"),
        chunksize=10,
    )

    for item in train_map:
        save_acous_to_file(item, "train", config)

    # preprocess valid files
    partial_fn = partial(gen_acous_features, config=config)
    valid_map = p.imap_unordered(
        partial_fn,
        tqdm(valid_iterator_data, total=len(valid_split), desc="[Preprocessing valid]"),
        chunksize=10,
    )
    for item in valid_map:
        save_acous_to_file(item, "valid", config)

    """
        sample = {
            "speaker_name": spkname,
            "filename"    : filename,
            "wav_path"    : wav_path,
            "text_ids"    : text_ids,
            "durs"        : durs,
            "embed_path"  : embed_path,
            "rate"        : self.target_rate,
        }
    """
def gen_acous_features(item, config):
    text_ids = item["text_ids"]
    durs = item["durs"]
    assert len(text_ids) == len(durs)

    global _feats_handle
    if _feats_handle is None:
        _feats_handle = AudioMelSpec(**config["feat_params"])

    audio = _feats_handle.load_wav(item["wav_path"])
    mel = _feats_handle.melspectrogram(audio)

    assert len(mel) == sum(durs)

    item["text_ids"] = np.array(text_ids)
    item["durs"]     = np.array(durs)
    item["mels"]     = mel
    item["embeds"]   = np.load(item["embed_path"])

    return item

def save_acous_to_file(features, subdir, config):
    filename = features["filename"]

    if config["format"] == "npy":
        save_list = [
            (features["text_ids"],  "ids",              "ids",              np.int32),
            (features["durs"],      "raw-durations",    "raw-durations",    np.int32),
            (features["mels"],      "raw-mels",         "raw-mels",         np.float32),
            (features["embeds"],    "embeds",           "embeds",           np.float32),
        ]
        for item, name_dir, name_file, fmt in save_list:
            np.save(
                os.path.join(
                    config["outdir"], subdir, name_dir, f"{filename}-{name_file}.npy"
                ),
                item.astype(fmt),
                allow_pickle=False,
            )
    else:
        raise ValueError("'npy' is the only supported format.")

'''
###############################################################################
################################ Vocoder ######################################
###############################################################################
'''
def preprocess_vocoder():
    """Run preprocessing process and compute statistics for normalizing."""
    config = parse_and_config()

    dataset_processor = {
        "multispk_voiceclone": MultiSPKVoiceCloneProcessor,
    }

    dataset_symbol = {
        "multispk_voiceclone": AISHELL_CHN_SYMBOLS,
    }

    dataset_cleaner = {
        "multispk_voiceclone": None,
    }

    logging.info(f"Selected '{config['dataset']}' processor.")
    processor = dataset_processor[config["dataset"]](
        config["rootdir"],
        symbols       = dataset_symbol[config["dataset"]],
        cleaner_names = dataset_cleaner[config["dataset"]],
        during_train  = True if config["during_train"] else False,
        mfaed_txt     = config["mfaed_txt"],
        wavs_dir      = config["wavs_dir"],
        embed_dir     = config["embed_dir"],
        spkinfo_dir   = config["spkinfo_dir"]
    )

    # check output directories
    build_dir = lambda x: [
        os.makedirs(os.path.join(config["outdir"], x, y), exist_ok=True)
        for y in ["norm-feats", "wavs"]
    ]
    build_dir("train")
    build_dir("valid")

    # save pretrained-processor to feature dir
    processor._save_mapper(
        os.path.join(config["outdir"], f"{config['dataset']}_mapper.json"),
        extra_attrs_to_save={"pinyin_dict": processor.pinyin_dict}
        if config["dataset"] == "multispk_voiceclone" else {},
    )

    # build train test split
    _Y = [i[0] for i in processor.items]
    train_split, valid_split = train_test_split(
        processor.items,
        test_size=config["test_size"],
        random_state=42,
        shuffle=True,
        stratify=_Y
    )
    logging.info(f"Training items: {len(train_split)}")
    logging.info(f"Validation items: {len(valid_split)}")

    train_utt_ids = [x[1] for x in train_split]
    valid_utt_ids = [x[1] for x in valid_split]

    # save train and valid utt_ids to track later
    np.save(os.path.join(config["outdir"], "train_utt_ids.npy"), train_utt_ids, allow_pickle=False)
    np.save(os.path.join(config["outdir"], "valid_utt_ids.npy"), valid_utt_ids, allow_pickle=False)

    # config["none_pinyin_symnum"] = processor.none_pinyin_symnum

    # define map iterator
    def iterator_data(items_list):
        for item in items_list:
            yield processor.get_one_sample(item)

    train_iterator_data = iterator_data(train_split)
    valid_iterator_data = iterator_data(valid_split)

    p = Pool(config["n_cpus"])

    # preprocess train files and get statistics for normalizing
    partial_fn = partial(gen_vocoder, config=config)
    train_map = p.imap_unordered(
        partial_fn,
        tqdm(train_iterator_data, total=len(train_split), desc="[Preprocessing train]"),
        chunksize=10,
    )

    for item in train_map:
        save_vocoder_to_file(item, "train", config)

    # preprocess valid files
    partial_fn = partial(gen_vocoder, config=config)
    valid_map = p.imap_unordered(
        partial_fn,
        tqdm(valid_iterator_data, total=len(valid_split), desc="[Preprocessing valid]"),
        chunksize=10,
    )
    for item in valid_map:
        save_vocoder_to_file(item, "valid", config)

    """
        sample = {
            "speaker_name": spkname,
            "filename"    : filename,
            "wav_path"    : wav_path,
            "text_ids"    : text_ids,
            "durs"        : durs,
            "embed_path"  : embed_path,
            "rate"        : self.target_rate,
        }
    """
def gen_vocoder(item, config):
    global _feats_handle
    if _feats_handle is None:
        _feats_handle = AudioMelSpec(**config["feat_params"])

    audio = _feats_handle.load_wav(item["wav_path"])
    mel = _feats_handle.melspectrogram(audio)

    # check audio and feature length
    audio = np.pad(audio, (0, _feats_handle.n_fft), mode="edge")
    audio = audio[: len(mel) * _feats_handle.hop_size]
    assert len(mel) * _feats_handle.hop_size == len(audio)

    item["audio"] = audio
    item["mels"]  = mel

    return item

def save_vocoder_to_file(features, subdir, config):
    filename = features["filename"]

    if config["format"] == "npy":
        save_list = [
            (features["audio"], "wavs",       "wave",         np.float32),
            (features["mels"],  "norm-feats", "norm-feats",   np.float32),
        ]
        for item, name_dir, name_file, fmt in save_list:
            np.save(
                os.path.join(
                    config["outdir"], subdir, name_dir, f"{filename}-{name_file}.npy"
                ),
                item.astype(fmt),
                allow_pickle=False,
            )
    else:
        raise ValueError("'npy' is the only supported format.")