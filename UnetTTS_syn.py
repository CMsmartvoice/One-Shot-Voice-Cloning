import re
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import yaml
from tensorflow_tts.audio_process import preprocess_wav
from tensorflow_tts.audio_process.audio_spec import AudioMelSpec
from tensorflow_tts.inference import AutoConfig, AutoProcessor, TFAutoModel


class UnetTTS():
    def __init__(self, models_and_params, text2id_mapper, feats_yaml):
        self.models_and_params      = models_and_params
        self.text2id_mapper         = text2id_mapper
        self.feats_yaml             = feats_yaml
        self.rhythm_txt_pat         = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^'^\-^#\d]")
        self.duration_stats_default = np.array([8.7, 2.8, 10.4, 4.7])
        self.text_id_start          = [1, 17, 79, 12, 49]
        self.text_id_end            = [25, 35, 13, 90, 1]
        self.phone_dur_min          = 5
        self.phone_dur_max          = 20
        self.__init_models()

    def one_shot_TTS(self, text, src_audio, duration_stats=None, is_wrap_txt=True):
        char_ids = self.txt2ids(text)
        # print(char_ids)

        mel_src = self.mel_feats_extractor(src_audio)

        if duration_stats is None:
            print("The statistics of the reference speech duration is calculated using the Style_Encoder.")
            duration_stats = self.infer_duration_stats(mel_src)
            print("Duration statistics equal to {}".format(duration_stats))
        elif len(duration_stats) != 4:
            print("Warning: The dimension of the reference speech duration'statistics of is not equal to 4, use default.")
            duration_stats = self.duration_stats_default

        if is_wrap_txt:
            char_ids = self.text_id_start + char_ids + self.text_id_end

        dur_pred = self.duration_model.inference(
            char_ids      = tf.expand_dims(tf.convert_to_tensor(char_ids, dtype=tf.int32), 0),
            duration_stat = tf.expand_dims(tf.convert_to_tensor(duration_stats, dtype=tf.float32), 0)
        )
        dur_gts = np.round(dur_pred[0].numpy()).astype(np.int32)

        mel_pred, _, _ = self.acous_model.inference(
                char_ids     = tf.expand_dims(tf.convert_to_tensor(char_ids, dtype=tf.int32), 0),
                duration_gts = tf.expand_dims(tf.convert_to_tensor(dur_gts, dtype=tf.int32), 0),
                mel_src      = tf.expand_dims(tf.convert_to_tensor(mel_src, dtype=tf.float32), 0)
        )

        if is_wrap_txt:
            start_dur = sum([dur_gts[i] for i in range(len(self.text_id_start))])
            end_dur = sum([dur_gts[-i] for i in range(1, len(self.text_id_end)+1)])
            audio = self.vocoder_model.inference(mel_pred[:, start_dur:-end_dur, :])[0, :, 0].numpy()

            return audio, mel_pred.numpy()[0][start_dur:-end_dur], mel_src
        else:
            audio = self.vocoder_model.inference(mel_pred)[0, :, 0].numpy()

            return audio, mel_pred.numpy()[0], mel_src

    def __init_models(self):
        self.processor = AutoProcessor.from_pretrained(pretrained_path=self.text2id_mapper)
        self.feats_config = yaml.load(open(self.feats_yaml), Loader=yaml.Loader)
        self.feats_handle = AudioMelSpec(**self.feats_config["feat_params"])
#         print(self.feats_config)

        self.duration_model = TFAutoModel.from_pretrained(config=AutoConfig.from_pretrained(self.models_and_params["duration_param"]), 
                                      pretrained_path=self.models_and_params["duration_model"],
                                      name="Normalized_duration_predictor")
        print("duration model load finished.")

        self.acous_model = TFAutoModel.from_pretrained(config=AutoConfig.from_pretrained(self.models_and_params["acous_param"]), 
                                  pretrained_path=self.models_and_params["acous_model"],
                                  name="Unet-TTS")
        print("acoustics model load finished.")

        self.vocoder_model = TFAutoModel.from_pretrained(config=AutoConfig.from_pretrained(self.models_and_params["vocoder_param"]),
                                pretrained_path=self.models_and_params["vocoder_model"],
                                name="Mb_MelGan")
        print("vocode model load finished.")

    def _stats_duration(self, dur_pos_embed):
        dur_pos_embed = dur_pos_embed[0].numpy()
        embed_num = dur_pos_embed.shape[-1] # 4

        mean = []
        std = []
        for i in range(embed_num):
            dur_pred = []
            phone_num = 0
            last = dur_pos_embed[1:, i][0]

            for j in dur_pos_embed[2:-1, i]:
                phone_num += 1
                if (phone_num >= self.phone_dur_max) or \
                    (i <= 1 and j > last and phone_num >= self.phone_dur_min) or \
                    (i > 1 and j < last and phone_num >= self.phone_dur_min):
                    dur_pred.append(phone_num)
                    phone_num = 0

                last = j

            mean.append(np.mean(dur_pred))
            std.append(np.std(dur_pred))

        return np.mean(mean)*1.0, np.mean(std)*0.8, np.mean(mean)*1.2, np.mean(std)*1.5

    def mel_feats_extractor(self, audio):
        return self.feats_handle.melspectrogram(audio)

    def txt2ids(self, input_text):
        assert re.search(self.rhythm_txt_pat, input_text) == None, "Remove punctuation"

        input_ids = self.processor.text_to_sequence(input_text, inference=True)
        return input_ids
    
    def infer_duration_stats(self, mel_src):
        dur_pos_embed = self.acous_model.extract_dur_pos_embed(
            tf.expand_dims(tf.convert_to_tensor(mel_src, dtype=tf.float32), 0)
        )
        return self._stats_duration(dur_pos_embed)

if __name__ == '__main__':
    """
    More examples can be seen in notebook.
    """

    models_and_params = {"duration_param": "train/configs/unetts_duration.yaml",
                        "duration_model": "models/duration4k.h5",
                        "acous_param": "train/configs/unetts_acous.yaml",
                        "acous_model": "models/acous12k.h5",
                        "vocoder_param": "train/configs/multiband_melgan.yaml",
                        "vocoder_model": "models/vocoder800k.h5"}

    feats_yaml = "train/configs/unetts_preprocess.yaml"

    text2id_mapper = "models/unetts_mapper.json"

    emotional_src_wav = {"neutral":{"wav": "test_wavs/neutral.wav",
                                    "dur_stat": "test_wavs/neutral_dur_stat.npy",
                                    "text": "现在全城的人都要向我借钱了"},
                        "happy": {"wav": "test_wavs/happy.wav",
                                    "dur_stat": "test_wavs/happy_dur_stat.npy",
                                    "text": "我参加了一个有关全球变暖的集会"},
                        "surprise": {"wav": "test_wavs/surprise.wav",
                                    "dur_stat": "test_wavs/surprise_dur_stat.npy",
                                    "text": "沙尘暴好像给每个人都带来了麻烦"},
                        "angry": {"wav": "test_wavs/angry.wav",
                                    "dur_stat": "test_wavs/angry_dur_stat.npy",
                                    "text": "不管怎么说主队好象是志在夺魁"},
                        "sad": {"wav": "test_wavs/sad.wav",
                                    "dur_stat": "test_wavs/sad_dur_stat.npy",
                                    "text": "我必须再次感谢您的慷慨相助"},
                        }

    Tts_handel = UnetTTS(models_and_params, text2id_mapper, feats_yaml)

    emotion_type = "neutral"
    # Inserting #3 marks into text is regarded as punctuation, and synthetic speech can produce pause.
    text = emotional_src_wav[emotion_type]["text"]

    wav_fpath = Path(emotional_src_wav[emotion_type]["wav"])
    src_audio = preprocess_wav(wav_fpath, source_sr=16000, normalize=True, trim_silence=True, is_sil_pad=True,
                        vad_window_length=30,
                        vad_moving_average_width=1,
                        vad_max_silence_length=1)

    """
    * The phoneme duration statistis of reference speech are composed of the initial and vowel of Chinese Pinyin, 
        including their respective mean and standard deviation. They will scale and bias the predicted duration of 
        phonemes and control the speed style of speech.
    * dur_stat = [initial_mean, initial_std, vowel_mean, vowel_std],  like dur_stat = [10., 2., 8., 4.]
    * The value is the frame length, and the frame shift of this model is 200.
    * The accurate value of phoneme duration can be extracted by ASR, MFA and other tools, 
        or the approximate value can be estimated using Style_Encoder.
    """
    Using_Style_Encoder = True

    if Using_Style_Encoder:
        syn_audio, _, _ = Tts_handel.one_shot_TTS(text, src_audio)
    else:
        # or dur_stat = None, or dur_stat = np.array([10., 2., 8., 4.])
        dur_stat = np.load(emotional_src_wav[emotion_type]["dur_stat"])
        print("dur_stat:", dur_stat)

        syn_audio, _, _ = Tts_handel.one_shot_TTS(text, src_audio, dur_stat)

    sf.write("./syn.wav", syn_audio, 16000, subtype='PCM_16')