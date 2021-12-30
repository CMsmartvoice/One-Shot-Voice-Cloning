# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
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
"""Perform preprocessing and raw feature extraction for Aishell3 dataset."""

import os
import re
from typing import Dict, List, Union, Tuple, Any

from dataclasses import dataclass, field
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin
from tensorflow_tts.processor import BaseProcessor
from tqdm import tqdm

from g2p_en import G2p
g2p = G2p()

_pad = ["pad"]
_eos = [None]

_pause = ["sil", "#0", "#1", "#3"]

CHN_WORD = "#0"
ENG_WORD = "#1"
PUC_SYM  = "#3"

_initials = [
    "^",
    "b",
    "c",
    "ch",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "x",
    "z",
    "zh",
]

_tones = ["1", "2", "3", "4", "5"]

_finals = [
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "i",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "ii",
    "iii",
    "in",
    "ing",
    "iong",
    "iou",
    "o",
    "ong",
    "ou",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "uei",
    "uen",
    "ueng",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
]

AISHELL_CHN_SYMBOLS = _pad + _pause + _initials + [i + j for i in _finals for j in _tones]


PINYIN_DICT = {
    "a": ("^", "a"),
    "ai": ("^", "ai"),
    "an": ("^", "an"),
    "ang": ("^", "ang"),
    "ao": ("^", "ao"),
    "ba": ("b", "a"),
    "bai": ("b", "ai"),
    "ban": ("b", "an"),
    "bang": ("b", "ang"),
    "bao": ("b", "ao"),
    "be": ("b", "e"),
    "bei": ("b", "ei"),
    "ben": ("b", "en"),
    "beng": ("b", "eng"),
    "bi": ("b", "i"),
    "bian": ("b", "ian"),
    "biao": ("b", "iao"),
    "bie": ("b", "ie"),
    "bin": ("b", "in"),
    "bing": ("b", "ing"),
    "bo": ("b", "o"),
    "bu": ("b", "u"),
    "ca": ("c", "a"),
    "cai": ("c", "ai"),
    "can": ("c", "an"),
    "cang": ("c", "ang"),
    "cao": ("c", "ao"),
    "ce": ("c", "e"),
    "cen": ("c", "en"),
    "ceng": ("c", "eng"),
    "cha": ("ch", "a"),
    "chai": ("ch", "ai"),
    "chan": ("ch", "an"),
    "chang": ("ch", "ang"),
    "chao": ("ch", "ao"),
    "che": ("ch", "e"),
    "chen": ("ch", "en"),
    "cheng": ("ch", "eng"),
    "chi": ("ch", "iii"),
    "chong": ("ch", "ong"),
    "chou": ("ch", "ou"),
    "chu": ("ch", "u"),
    "chua": ("ch", "ua"),
    "chuai": ("ch", "uai"),
    "chuan": ("ch", "uan"),
    "chuang": ("ch", "uang"),
    "chui": ("ch", "uei"),
    "chun": ("ch", "uen"),
    "chuo": ("ch", "uo"),
    "ci": ("c", "ii"),
    "cong": ("c", "ong"),
    "cou": ("c", "ou"),
    "cu": ("c", "u"),
    "cuan": ("c", "uan"),
    "cui": ("c", "uei"),
    "cun": ("c", "uen"),
    "cuo": ("c", "uo"),
    "da": ("d", "a"),
    "dai": ("d", "ai"),
    "dan": ("d", "an"),
    "dang": ("d", "ang"),
    "dao": ("d", "ao"),
    "de": ("d", "e"),
    "dei": ("d", "ei"),
    "den": ("d", "en"),
    "deng": ("d", "eng"),
    "di": ("d", "i"),
    "dia": ("d", "ia"),
    "dian": ("d", "ian"),
    "diao": ("d", "iao"),
    "die": ("d", "ie"),
    "ding": ("d", "ing"),
    "diu": ("d", "iou"),
    "dong": ("d", "ong"),
    "dou": ("d", "ou"),
    "du": ("d", "u"),
    "duan": ("d", "uan"),
    "dui": ("d", "uei"),
    "dun": ("d", "uen"),
    "duo": ("d", "uo"),
    "e": ("^", "e"),
    "ei": ("^", "ei"),
    "en": ("^", "en"),
    "ng": ("^", "en"),
    "eng": ("^", "eng"),
    "er": ("^", "er"),
    "fa": ("f", "a"),
    "fan": ("f", "an"),
    "fang": ("f", "ang"),
    "fei": ("f", "ei"),
    "fen": ("f", "en"),
    "feng": ("f", "eng"),
    "fo": ("f", "o"),
    "fou": ("f", "ou"),
    "fu": ("f", "u"),
    "ga": ("g", "a"),
    "gai": ("g", "ai"),
    "gan": ("g", "an"),
    "gang": ("g", "ang"),
    "gao": ("g", "ao"),
    "ge": ("g", "e"),
    "gei": ("g", "ei"),
    "gen": ("g", "en"),
    "geng": ("g", "eng"),
    "gong": ("g", "ong"),
    "gou": ("g", "ou"),
    "gu": ("g", "u"),
    "gua": ("g", "ua"),
    "guai": ("g", "uai"),
    "guan": ("g", "uan"),
    "guang": ("g", "uang"),
    "gui": ("g", "uei"),
    "gun": ("g", "uen"),
    "guo": ("g", "uo"),
    "ha": ("h", "a"),
    "hai": ("h", "ai"),
    "han": ("h", "an"),
    "hang": ("h", "ang"),
    "hao": ("h", "ao"),
    "he": ("h", "e"),
    "hei": ("h", "ei"),
    "hen": ("h", "en"),
    "heng": ("h", "eng"),
    "hong": ("h", "ong"),
    "hou": ("h", "ou"),
    "hu": ("h", "u"),
    "hua": ("h", "ua"),
    "huai": ("h", "uai"),
    "huan": ("h", "uan"),
    "huang": ("h", "uang"),
    "hui": ("h", "uei"),
    "hun": ("h", "uen"),
    "huo": ("h", "uo"),
    "ji": ("j", "i"),
    "jia": ("j", "ia"),
    "jian": ("j", "ian"),
    "jiang": ("j", "iang"),
    "jiao": ("j", "iao"),
    "jie": ("j", "ie"),
    "jin": ("j", "in"),
    "jing": ("j", "ing"),
    "jiong": ("j", "iong"),
    "jiu": ("j", "iou"),
    "ju": ("j", "v"),
    "juan": ("j", "van"),
    "jue": ("j", "ve"),
    "jun": ("j", "vn"),
    "ka": ("k", "a"),
    "kai": ("k", "ai"),
    "kan": ("k", "an"),
    "kang": ("k", "ang"),
    "kao": ("k", "ao"),
    "ke": ("k", "e"),
    "kei": ("k", "ei"),
    "ken": ("k", "en"),
    "keng": ("k", "eng"),
    "kong": ("k", "ong"),
    "kou": ("k", "ou"),
    "ku": ("k", "u"),
    "kua": ("k", "ua"),
    "kuai": ("k", "uai"),
    "kuan": ("k", "uan"),
    "kuang": ("k", "uang"),
    "kui": ("k", "uei"),
    "kun": ("k", "uen"),
    "kuo": ("k", "uo"),
    "la": ("l", "a"),
    "lai": ("l", "ai"),
    "lan": ("l", "an"),
    "lang": ("l", "ang"),
    "lao": ("l", "ao"),
    "le": ("l", "e"),
    "lei": ("l", "ei"),
    "leng": ("l", "eng"),
    "li": ("l", "i"),
    "lia": ("l", "ia"),
    "lian": ("l", "ian"),
    "liang": ("l", "iang"),
    "liao": ("l", "iao"),
    "lie": ("l", "ie"),
    "lin": ("l", "in"),
    "ling": ("l", "ing"),
    "liu": ("l", "iou"),
    "lo": ("l", "o"),
    "long": ("l", "ong"),
    "lou": ("l", "ou"),
    "lu": ("l", "u"),
    "lv": ("l", "v"),
    "luan": ("l", "uan"),
    "lve": ("l", "ve"),
    "lue": ("l", "ve"),
    "lun": ("l", "uen"),
    "luo": ("l", "uo"),
    "ma": ("m", "a"),
    "mai": ("m", "ai"),
    "man": ("m", "an"),
    "mang": ("m", "ang"),
    "mao": ("m", "ao"),
    "me": ("m", "e"),
    "mei": ("m", "ei"),
    "men": ("m", "en"),
    "meng": ("m", "eng"),
    "mi": ("m", "i"),
    "mian": ("m", "ian"),
    "miao": ("m", "iao"),
    "mie": ("m", "ie"),
    "min": ("m", "in"),
    "ming": ("m", "ing"),
    "miu": ("m", "iou"),
    "mo": ("m", "o"),
    "mou": ("m", "ou"),
    "mu": ("m", "u"),
    "na": ("n", "a"),
    "nai": ("n", "ai"),
    "nan": ("n", "an"),
    "nang": ("n", "ang"),
    "nao": ("n", "ao"),
    "ne": ("n", "e"),
    "nei": ("n", "ei"),
    "nen": ("n", "en"),
    "neng": ("n", "eng"),
    "ni": ("n", "i"),
    "nia": ("n", "ia"),
    "nian": ("n", "ian"),
    "niang": ("n", "iang"),
    "niao": ("n", "iao"),
    "nie": ("n", "ie"),
    "nin": ("n", "in"),
    "ning": ("n", "ing"),
    "niu": ("n", "iou"),
    "nong": ("n", "ong"),
    "nou": ("n", "ou"),
    "nu": ("n", "u"),
    "nv": ("n", "v"),
    "nuan": ("n", "uan"),
    "nve": ("n", "ve"),
    "nue": ("n", "ve"),
    "nuo": ("n", "uo"),
    "o": ("^", "o"),
    "ou": ("^", "ou"),
    "pa": ("p", "a"),
    "pai": ("p", "ai"),
    "pan": ("p", "an"),
    "pang": ("p", "ang"),
    "pao": ("p", "ao"),
    "pe": ("p", "e"),
    "pei": ("p", "ei"),
    "pen": ("p", "en"),
    "peng": ("p", "eng"),
    "pi": ("p", "i"),
    "pian": ("p", "ian"),
    "piao": ("p", "iao"),
    "pie": ("p", "ie"),
    "pin": ("p", "in"),
    "ping": ("p", "ing"),
    "po": ("p", "o"),
    "pou": ("p", "ou"),
    "pu": ("p", "u"),
    "qi": ("q", "i"),
    "qia": ("q", "ia"),
    "qian": ("q", "ian"),
    "qiang": ("q", "iang"),
    "qiao": ("q", "iao"),
    "qie": ("q", "ie"),
    "qin": ("q", "in"),
    "qing": ("q", "ing"),
    "qiong": ("q", "iong"),
    "qiu": ("q", "iou"),
    "qu": ("q", "v"),
    "quan": ("q", "van"),
    "que": ("q", "ve"),
    "qun": ("q", "vn"),
    "ran": ("r", "an"),
    "rang": ("r", "ang"),
    "rao": ("r", "ao"),
    "re": ("r", "e"),
    "ren": ("r", "en"),
    "reng": ("r", "eng"),
    "ri": ("r", "iii"),
    "rong": ("r", "ong"),
    "rou": ("r", "ou"),
    "ru": ("r", "u"),
    "rua": ("r", "ua"),
    "ruan": ("r", "uan"),
    "rui": ("r", "uei"),
    "run": ("r", "uen"),
    "ruo": ("r", "uo"),
    "sa": ("s", "a"),
    "sai": ("s", "ai"),
    "san": ("s", "an"),
    "sang": ("s", "ang"),
    "sao": ("s", "ao"),
    "se": ("s", "e"),
    "sen": ("s", "en"),
    "seng": ("s", "eng"),
    "sha": ("sh", "a"),
    "shai": ("sh", "ai"),
    "shan": ("sh", "an"),
    "shang": ("sh", "ang"),
    "shao": ("sh", "ao"),
    "she": ("sh", "e"),
    "shei": ("sh", "ei"),
    "shen": ("sh", "en"),
    "sheng": ("sh", "eng"),
    "shi": ("sh", "iii"),
    "shou": ("sh", "ou"),
    "shu": ("sh", "u"),
    "shua": ("sh", "ua"),
    "shuai": ("sh", "uai"),
    "shuan": ("sh", "uan"),
    "shuang": ("sh", "uang"),
    "shui": ("sh", "uei"),
    "shun": ("sh", "uen"),
    "shuo": ("sh", "uo"),
    "si": ("s", "ii"),
    "song": ("s", "ong"),
    "sou": ("s", "ou"),
    "su": ("s", "u"),
    "suan": ("s", "uan"),
    "sui": ("s", "uei"),
    "sun": ("s", "uen"),
    "suo": ("s", "uo"),
    "ta": ("t", "a"),
    "tai": ("t", "ai"),
    "tan": ("t", "an"),
    "tang": ("t", "ang"),
    "tao": ("t", "ao"),
    "te": ("t", "e"),
    "tei": ("t", "ei"),
    "teng": ("t", "eng"),
    "ti": ("t", "i"),
    "tian": ("t", "ian"),
    "tiao": ("t", "iao"),
    "tie": ("t", "ie"),
    "ting": ("t", "ing"),
    "tong": ("t", "ong"),
    "tou": ("t", "ou"),
    "tu": ("t", "u"),
    "tuan": ("t", "uan"),
    "tui": ("t", "uei"),
    "tun": ("t", "uen"),
    "tuo": ("t", "uo"),
    "wa": ("^", "ua"),
    "wai": ("^", "uai"),
    "wan": ("^", "uan"),
    "wang": ("^", "uang"),
    "wei": ("^", "uei"),
    "wen": ("^", "uen"),
    "weng": ("^", "ueng"),
    "wo": ("^", "uo"),
    "wu": ("^", "u"),
    "xi": ("x", "i"),
    "xia": ("x", "ia"),
    "xian": ("x", "ian"),
    "xiang": ("x", "iang"),
    "xiao": ("x", "iao"),
    "xie": ("x", "ie"),
    "xin": ("x", "in"),
    "xing": ("x", "ing"),
    "xiong": ("x", "iong"),
    "xiu": ("x", "iou"),
    "xu": ("x", "v"),
    "xuan": ("x", "van"),
    "xue": ("x", "ve"),
    "xun": ("x", "vn"),
    "ya": ("^", "ia"),
    "yan": ("^", "ian"),
    "yang": ("^", "iang"),
    "yao": ("^", "iao"),
    "ye": ("^", "ie"),
    "yi": ("^", "i"),
    "yin": ("^", "in"),
    "ying": ("^", "ing"),
    "yo": ("^", "iou"),
    "yong": ("^", "iong"),
    "you": ("^", "iou"),
    "yu": ("^", "v"),
    "yuan": ("^", "van"),
    "yue": ("^", "ve"),
    "yun": ("^", "vn"),
    "za": ("z", "a"),
    "zai": ("z", "ai"),
    "zan": ("z", "an"),
    "zang": ("z", "ang"),
    "zao": ("z", "ao"),
    "ze": ("z", "e"),
    "zei": ("z", "ei"),
    "zen": ("z", "en"),
    "zeng": ("z", "eng"),
    "zha": ("zh", "a"),
    "zhai": ("zh", "ai"),
    "zhan": ("zh", "an"),
    "zhang": ("zh", "ang"),
    "zhao": ("zh", "ao"),
    "zhe": ("zh", "e"),
    "zhei": ("zh", "ei"),
    "zhen": ("zh", "en"),
    "zheng": ("zh", "eng"),
    "zhi": ("zh", "iii"),
    "zhong": ("zh", "ong"),
    "zhou": ("zh", "ou"),
    "zhu": ("zh", "u"),
    "zhua": ("zh", "ua"),
    "zhuai": ("zh", "uai"),
    "zhuan": ("zh", "uan"),
    "zhuang": ("zh", "uang"),
    "zhui": ("zh", "uei"),
    "zhun": ("zh", "uen"),
    "zhuo": ("zh", "uo"),
    "zi": ("z", "ii"),
    "zong": ("z", "ong"),
    "zou": ("z", "ou"),
    "zu": ("z", "u"),
    "zuan": ("z", "uan"),
    "zui": ("z", "uei"),
    "zun": ("z", "uen"),
    "zuo": ("z", "uo"),
}

zh_pattern = re.compile(r"([\u4e00-\u9fa5]+)")
en_pattern = re.compile(r"([a-zA-Z]+)")

def is_zh(word):
    global zh_pattern
    match = zh_pattern.search(word)
    return match is not None

def is_en(word):
    global en_pattern
    match = en_pattern.search(word)
    return match is not None

class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


@dataclass
class MultiSPKVoiceCloneProcessor(BaseProcessor):

    pinyin_dict        : Dict[str, Tuple[str, str]] = field(default_factory=lambda: PINYIN_DICT)
    cleaner_names      : str                        = None
    target_rate        : int                        = 16000
    speaker_name       : str                        = "multispk_voiceclone"
    none_pinyin_symnum : int                        = len(_pad + _pause)
    during_train       : bool                       = False
    f0_train           : bool                       = False
    all_train          : bool                       = False
    mfaed_txt          : str                        = ""
    wavs_dir           : str                        = ""
    embed_dir          : str                        = ""
    spkinfo_dir        : str                        = ""
    unseen_dir         : str                        = ""

    def __post_init__(self):
        self.pinyin_parser = self.get_pinyin_parser()

        if self.spkinfo_dir:
            self.create_speaker_info()

        if self.unseen_dir:
            self.create_unseen_speaker()

        super().__post_init__()

    def setup_eos_token(self):
        return _eos[0]

    def create_speaker_info(self):
        self.spk2sex_dict = {}
        with open(self.spkinfo_dir, "r") as fr:
            lines = fr.readlines()

        for line in lines:
            spk_name, _, sex, *_ = line.strip().split()
            if spk_name not in self.spk2sex_dict.keys():
                self.spk2sex_dict[spk_name] = sex
        
        print("*"*50)
        print(f"Have {len(self.spk2sex_dict)} Speakers")
        print("*"*50)

    def create_unseen_speaker(self):
        self.unseen_spk = []
        with open(self.unseen_dir, "r") as fr:
            lines = fr.readlines()

        for line in lines:
            self.unseen_spk.append(line.strip())
        
        self.unseen_spk_num = 0

        print("*"*50)
        print(f"Have {len(self.unseen_spk)} UNSeen Speakers")
        print("*"*50)

    # TODO now just support for aishell3
    def create_items(self):
        items = []
        if self.data_dir:
            with open(
                os.path.join(self.data_dir, self.mfaed_txt),
                encoding="utf-8",
            ) as ttf:
                lines = ttf.readlines()
                for line in tqdm(lines):
                    filename, phoneseq, durseq = line.strip().split("|")
                    spkname = filename[:7] if filename[0] == "S" else filename.split("_")[0]

                    if self.spkinfo_dir and self.spk2sex_dict[spkname] == "male":
                        continue

                    # if self.unseen_dir and (spkname in self.unseen_spk or "_" in filename):
                    if self.unseen_dir and spkname in self.unseen_spk:
                        self.unseen_spk_num += 1
                        continue

                    wav_path = os.path.join(self.data_dir, self.wavs_dir, f"{spkname}", f"{filename}.wav")
                    embed_path = os.path.join(self.data_dir, self.embed_dir, f"{filename}-embed.npy")

                    if ((self.during_train or self.f0_train) and os.path.exists(wav_path)) or \
                        (self.all_train and os.path.exists(wav_path) and os.path.exists(embed_path)):
                        try:
                            text_ids = [self.symbol_to_id[phone] for phone in phoneseq.split()]
                            durs = [int(dur) for dur in durseq.split()]
                            embed_path = embed_path if self.all_train else None
                            assert len(text_ids) == len(durs)
                        except Exception:
                            print("error: generate sequence ids", filename)
                            continue
                    
                        items.append([spkname, filename, wav_path, text_ids, durs, embed_path])

            self.items = items
            print("*"*50)
            print(f"Have {len(self.items)} samples")
            if self.unseen_dir:
                print(f"Have {self.unseen_spk_num} UNSeen samples")
            print("*"*50)

    def get_phoneme_from_char_and_pinyin(self, txt, pinyin):
        txt = txt.replace("'", "")
        txt = txt.replace("-", "")

        phrase_list   = re.split("#\d", txt)
        rhythms_list  = re.findall("#\d", txt)

        pinyin_index = 0
        last_phrase_type = 'chn'

        result = ["sil"]

        phrase = phrase_list[0]

        if is_zh(phrase):
            length = len(phrase)

            for pinyin_one in pinyin[pinyin_index:pinyin_index+length]:
                tone = pinyin_one[-1]
                a1, a2 = self.pinyin_dict[pinyin_one[:-1]]
                result.append(a1)
                result.append(a2+tone)
            
            pinyin_index += length

            last_phrase_type = 'chn'
        
        elif is_en(phrase):
            pinyin_one = pinyin[pinyin_index]

            result += [ph[:-1] if ph[-1].isdigit() else ph for ph in pinyin_one.split("*")]

            pinyin_index += 1

            last_phrase_type = 'eng'

        # TODO
        elif phrase == "" or phrase == " ":
            pass

        else:
            print("Error: processed text ->", txt)
            assert False

        phrase_list = phrase_list[1:]

        for rhythm, phrase in zip(rhythms_list, phrase_list):

            if is_zh(phrase):
                if rhythm == "#3":
                    result.append(PUC_SYM)
                elif last_phrase_type == 'eng':
                    result.append(ENG_WORD)
                else:
                    result.append(CHN_WORD)

                length = len(phrase)

                for pinyin_one in pinyin[pinyin_index:pinyin_index+length]:
                    tone = pinyin_one[-1]
                    a1, a2 = self.pinyin_dict[pinyin_one[:-1]]
                    result.append(a1)
                    result.append(a2+tone)
                
                pinyin_index += length

                last_phrase_type = 'chn'
            
            elif is_en(phrase):
                if rhythm == "#3":
                    result.append(PUC_SYM)
                else:
                    result.append(ENG_WORD)

                pinyin_one = pinyin[pinyin_index]

                result += [ph[:-1] if ph[-1].isdigit() else ph for ph in pinyin_one.split("*")]

                pinyin_index += 1

                last_phrase_type = 'eng'

            # TODO
            elif phrase == "" or phrase == " ":
                continue

            else:
                print("Error: processed text ->", txt)
                assert False

        # result.append(PUC_SYM)
        result.append("sil")

        assert pinyin_index == len(pinyin)

        return result

    def get_one_sample(self, item): 
        spkname, filename, wav_path, text_ids, durs, embed_path = item

        sample = {
            "speaker_name": spkname,
            "filename"    : filename,
            "wav_path"    : wav_path,
            "text_ids"    : text_ids,
            "durs"        : durs,
            "embed_path"  : embed_path,
            "rate"        : self.target_rate,
        }

        return sample

    def get_pinyin_parser(self):
        my_pinyin = Pinyin(MyConverter())
        pinyin = my_pinyin.pinyin
        return pinyin

    def text_to_sequence(self, text, inference=False):
        if inference:
            pinyin = self.pinyin_parser(text, style=Style.TONE3)
            # print(pinyin)
            new_pinyin = []
            for x in pinyin:
                x = "".join(x)
                if "#" not in x:
                    new_pinyin.append(x)
                else:
                    if len(x) == 2:
                        continue
                    else:
                        eng_list = re.split("#\d", x)
                        if eng_list[0] == "":
                            eng_list = eng_list[1:]
                        if eng_list[-1] == "":
                            eng_list = eng_list[:-1]

                        for e in eng_list:
                            new_pinyin.append("*".join(g2p(e)))

            print(new_pinyin)

            phonemes = self.get_phoneme_from_char_and_pinyin(text, new_pinyin)
            text = " ".join(phonemes)
            print(f"phoneme seq: {text}")

        sequence = []
        #print("text",text)
        for symbol in text.split():
            idx = self.symbol_to_id[symbol]
            sequence.append(idx)

        return sequence

    def create_speaker_map(self):
        pass