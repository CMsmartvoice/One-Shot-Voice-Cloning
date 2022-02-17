## Unet-TTS: Improving Unseen Speaker and Style Transfer in One-shot Voice Cloning
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)

> 中文 | [English](README.md)

:exclamation: 提供推理代码和预训练模型，你可以生成想要的文本语音。

:star: 模型只在正常情绪的语料上训练，没有使用其他任何强烈情感的语料。

:star: 受到训练语料的限制，一般的说话人编码或者非监督风格学习方法都很难模仿未见过的语音。训练数据分布范围外的风格迁移仍具有很大的挑战。

:star: 依赖Unet网络和AdaIN层，我们的方法在未见风格上有很强的迁移能力。

[推理代码](notebook) or [在线notebook](https://colab.research.google.com/drive/1sEDvKTJCY7uosb7TvTqwyUdwNPiv3pBW#scrollTo=puzhCI99LY_a)

[Demo results](https://cmsmartvoice.github.io/Unet-TTS/)

[Paper link](https://arxiv.org/abs/2109.11115)

![](./pics/structure.png)

---
:star: 现在只需要输入一条参考语音就可以进行克隆TTS，而不再需要手动输入参考语音的时长统计信息。

:smile: 我们正在准备基于aishell3数据的训练流程，敬请期待。

流程包括:
- [x] 一句话语音克隆推理
- [x] 参考音频的时长统计信息可以有训练的Style_Encoder估计
- [ ] 基于说话人编码的多说话人TTS，它可以提供不错的Content Encoder
- [ ] Unet-TTS训练
- [ ] C++推理

---
### Install Requirements
- Install the appropriate TensorFlow and tensorflow-addons versions according to CUDA version. 
- The default is TensorFlow 2.6 and tensorflow-addons 0.14.0.
```shell
pip install TensorFlowTTS
```

### Usage
- see file UnetTTS_syn.py or notebook
```shell
CUDA_VISIBLE_DEVICES=0 python UnetTTS_syn.py
```

```python
from UnetTTS_syn import UnetTTS

models_and_params = {"duration_param": "train/configs/unetts_duration.yaml",
                    "duration_model": "models/duration4k.h5",
                    "acous_param": "train/configs/unetts_acous.yaml",
                    "acous_model": "models/acous12k.h5",
                    "vocoder_param": "train/configs/multiband_melgan.yaml",
                    "vocoder_model": "models/vocoder800k.h5"}

feats_yaml = "train/configs/unetts_preprocess.yaml"

text2id_mapper = "models/unetts_mapper.json"

Tts_handel = UnetTTS(models_and_params, text2id_mapper, feats_yaml)

# text: input text
# src_audio: reference audio
# The duration statistics of the reference speech can be estimated using Style_Encoder
syn_audio, _, _ = Tts_handel.one_shot_TTS(text, src_audio)

# OR
# dur_stat: phoneme duration statistis to control speed rate
syn_audio, _, _ = Tts_handel.one_shot_TTS(text, src_audio, dur_stat)
```

### Reference
https://github.com/TensorSpeech/TensorFlowTTS

https://github.com/CorentinJ/Real-Time-Voice-Cloning 