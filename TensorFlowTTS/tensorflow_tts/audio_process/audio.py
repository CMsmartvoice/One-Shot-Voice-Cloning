import struct
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
from scipy.ndimage.morphology import binary_dilation

try:
    import webrtcvad
except:
    print("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None


# ## Voice Activation Detection
# # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# # This sets the granularity of the VAD. Should not need to be changed.
# vad_window_length = 30  # In milliseconds
# # Number of frames to average together when performing the moving average smoothing.
# # The larger this value, the larger the VAD variations must be to not get smoothed out. 
# vad_moving_average_width = 8
# # Maximum number of consecutive silent frames a segment can have.
# vad_max_silence_length = 6

int16_max = (2 ** 15) - 1
sampling_rate = 16000

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True,
                   is_sil_pad: Optional[bool] = True,
                   vad_window_length = 30,
                   vad_moving_average_width = 8,
                   vad_max_silence_length = 6):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)

    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav)

    if trim_silence:
        wav = trim_long_silences(wav, vad_window_length, vad_moving_average_width, vad_max_silence_length)

    if is_sil_pad:
        wav = sil_pad(wav)

    return wav

def normalize_volume(wav, ratio=0.6):
    return wav / np.max(np.abs(wav)) * ratio

def sil_pad(wav, pad_length=100):
    pad_length = int(sampling_rate / 1000 * pad_length)
    return np.pad(wav, (pad_length, pad_length))

def trim_long_silences(wav, vad_window_length, vad_moving_average_width, vad_max_silence_length):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]

def melbasis_make(sr=16000, n_fft=1024, n_mels=80, fmin=80, fmax=7600):
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

def mel_make(filepath: str, sr=16000, n_fft=1024, framesize=256, mel_basis=None, fn=None):
    if fn is None:
        audio, _ = librosa.load(filepath, sr=sr)
    else:
        audio = fn(filepath, trim_silence=False, is_sil_pad=False)

    D = librosa.stft(audio, n_fft=n_fft, hop_length=framesize)
    S, _ = librosa.magphase(D)

    if mel_basis:
        mel = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10)).T
        return audio, mel
    else:
        return audio, S