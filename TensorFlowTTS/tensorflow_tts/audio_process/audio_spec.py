import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import soundfile as sf

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

class AudioMelSpec():
    '''
    Audio to Mel_Spec
    '''
    def __init__(
        self, sample_rate=16000, n_fft=800, num_mels=80, hop_size=200, win_size=800,
        fmin=55, fmax=7600, min_level_db=-100, ref_level_db=20, max_abs_value=4.,
        preemphasis=0.97, preemphasize=True,
        signal_normalization=True, allow_clipping_in_normalization=True, symmetric_mels=True,
        power=1.5, griffin_lim_iters=60,
        rescale=True, rescaling_max=0.9
    ):
        self.sample_rate   = sample_rate
        self.n_fft         = n_fft
        self.num_mels      = num_mels
        self.hop_size      = hop_size
        self.win_size      = win_size
        self.fmin          = fmin
        self.fmax          = fmax
        self.min_level_db  = min_level_db
        self.ref_level_db  = ref_level_db
        self.max_abs_value = max_abs_value
        self.preemphasis   = preemphasis
        self.preemphasize  = preemphasize

        self.signal_normalization            = signal_normalization
        self.symmetric_mels                  = symmetric_mels
        self.allow_clipping_in_normalization = allow_clipping_in_normalization

        self.power = power
        self.griffin_lim_iters = griffin_lim_iters

        self.rescale = rescale
        self.rescaling_max = rescaling_max

        self._mel_basis_create()

    def _mel_basis_create(self):
        self._mel_basis     = librosa.filters.mel(self.sample_rate, self.n_fft, self.num_mels, self.fmin, self.fmax)
        self._inv_mel_basis = np.linalg.pinv(self._mel_basis)

    def _stft(self, y):
        return librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.win_size)

    def _istft(self, y):
        return librosa.istft(y, hop_length=self.hop_size, win_length=self.win_size)

    def _linear_to_mel(self, spectogram):
        return np.dot(self._mel_basis, spectogram)

    def _mel_to_linear(self, mel_spectrogram):
        return np.maximum(1e-10, np.dot(self._inv_mel_basis, mel_spectrogram))

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, (x) * 0.05)

    def _normalize(self, S):
        if self.allow_clipping_in_normalization:
            if self.symmetric_mels:
                return np.clip((2 * self.max_abs_value) * ((S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value,
                            -self.max_abs_value, self.max_abs_value)
            else:
                return np.clip(self.max_abs_value * ((S - self.min_level_db) / (-self.min_level_db)), 0, self.max_abs_value)
        
        assert S.max() <= 0 and S.min() - self.min_level_db >= 0
        if self.symmetric_mels:
            return (2 * self.max_abs_value) * ((S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value
        else:
            return self.max_abs_value * ((S - self.min_level_db) / (-self.min_level_db))

    def _denormalize(self, D):
        if self.allow_clipping_in_normalization:
            if self.symmetric_mels:
                return (((np.clip(D, -self.max_abs_value,
                                self.max_abs_value) + self.max_abs_value) * -self.min_level_db / (2 * self.max_abs_value))
                        + self.min_level_db)
            else:
                return ((np.clip(D, 0, self.max_abs_value) * -self.min_level_db / self.max_abs_value) + self.min_level_db)
        
        if self.symmetric_mels:
            return (((D + self.max_abs_value) * -self.min_level_db / (2 * self.max_abs_value)) + self.min_level_db)
        else:
            return ((D * -self.min_level_db / self.max_abs_value) + self.min_level_db)

    def _griffin_lim(self, S):
        """librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        """
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def load_wav(self, wav_fpath):
        wav, _  = librosa.load(wav_fpath, sr=self.sample_rate)
        if self.rescale:
            wav = wav / np.abs(wav).max() * self.rescaling_max
        return wav

    def save_wav(self, wav, fpath):
        if self.rescale:
            wav = wav / np.abs(wav).max() * self.rescaling_max
        sf.write(fpath, wav, self.sample_rate, subtype="PCM_16")

    def melspectrogram(self, wav):
        D = self._stft(preemphasis(wav, self.preemphasis, self.preemphasize))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        
        if self.signal_normalization:
            return self._normalize(S.T)
        return S.T

    def inv_mel_spectrogram(self, mel_spectrogram):
        """Converts mel spectrogram to waveform using librosa"""
        if self.signal_normalization:
            D = self._denormalize(mel_spectrogram.T)
        else:
            D = mel_spectrogram.T
        
        S = self._mel_to_linear(self._db_to_amp(D + self.ref_level_db))  # Convert back to linear
        
        return inv_preemphasis(self._griffin_lim(S ** self.power), self.preemphasis, self.preemphasize)

    def compare_plot(self, targets, preds, filepath=None, frame_real_len=None, text=None):
        if frame_real_len:
            targets = targets[:frame_real_len]
            preds   = preds[:frame_real_len]

        fig = plt.figure(figsize=(14,10))

        if text:
            fig.text(0.4, 0.48, text, horizontalalignment="center", fontsize=16)

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        im = ax1.imshow(targets.T, aspect='auto', origin="lower", interpolation="none")
        ax1.set_title("Target Mel-Spectrogram")
        fig.colorbar(mappable=im, shrink=0.65, ax=ax1)
        
        im = ax2.imshow(preds.T, aspect='auto', origin="lower", interpolation="none")
        ax2.set_title("Pred Mel-Spectrogram")
        fig.colorbar(mappable=im, shrink=0.65, ax=ax2)

        plt.tight_layout()
        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath)
            plt.close()

    def melspec_plot(self, mels):
        plt.figure(figsize=(10,6))
        plt.imshow(mels.T, aspect='auto', origin="lower", interpolation="none")
        plt.colorbar()
        plt.show()

class AudioSpec():
    ''' # TODO
    Now just for sqrt(sp) from world
    '''
    def __init__(self, sr, nfft, mel_dim=80, f0_min=71, f0_max=7800,
                    min_level_db=-120., ref_level_db=-5., max_abs_value=4.,
                    is_norm=True, is_symmetric=True, is_clipping_in_normalization=False):
        self.sr      = sr
        self.nfft    = nfft
        self.mel_dim = mel_dim
        self.f0_min  = f0_min
        self.f0_max  = f0_max

        self.min_level_db  = min_level_db
        self.min_level_amp = np.exp((self.min_level_db + 0.1) / 20 * np.log(10))

        # sp from world, self.ref_level_db should be less than zero
        # otherwise, is_clipping_in_normalization should be true
        self.ref_level_db  = ref_level_db
        self.max_abs_value = max_abs_value
        
        self.is_norm                      = is_norm
        self.is_symmetric                 = is_symmetric
        self.is_clipping_in_normalization = is_clipping_in_normalization

        if self.ref_level_db > 0.:
            try:
                assert self.is_norm and self.is_clipping_in_normalization
            except:
                self.is_clipping_in_normalization = True

        self._mel_basis_create()

    def _mel_basis_create(self):
        self._mel_basis     = librosa.filters.mel(self.sr, self.nfft, self.mel_dim, self.f0_min, self.f0_max)
        self._inv_mel_basis = np.linalg.pinv(self._mel_basis)

    def _normalize(self, log_sepc, is_symmetric, is_clipping_in_normalization):
        if is_clipping_in_normalization:
            if is_symmetric:
                return np.clip((2 * self.max_abs_value) * ((log_sepc - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value,
                               -self.max_abs_value, self.max_abs_value)
            else:
                return np.clip(self.max_abs_value * ((log_sepc - self.min_level_db) / (-self.min_level_db)), 0, self.max_abs_value)

        assert log_sepc.max() <= 0 and log_sepc.min() >= self.min_level_db
        if is_symmetric:
            return (2 * self.max_abs_value) * ((log_sepc - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value
        else:
            return self.max_abs_value * ((log_sepc - self.min_level_db) / (-self.min_level_db))

    def _denormalize(self, log_sepc, is_symmetric, is_clipping_in_normalization):
        if is_clipping_in_normalization:
            if is_symmetric:
                return (((np.clip(log_sepc, -self.max_abs_value,
                                  self.max_abs_value) + self.max_abs_value) * -self.min_level_db / (2 * self.max_abs_value))
                        + self.min_level_db)
            else:
                return ((np.clip(log_sepc, 0, self.max_abs_value) * -self.min_level_db / self.max_abs_value) + self.min_level_db)

        if is_symmetric:
            return (((log_sepc + self.max_abs_value) * -self.min_level_db / (2 * self.max_abs_value)) + self.min_level_db)
        else:
            return ((log_sepc * -self.min_level_db / self.max_abs_value) + self.min_level_db)

    def ampspec2logspec(self, amp_spec):
        mel_spec = np.dot(amp_spec, self._mel_basis.T)
        log_sepc = 20 * np.log10(np.maximum(self.min_level_amp, mel_spec)) - self.ref_level_db

        if self.is_norm:
            log_sepc = self._normalize(log_sepc, self.is_symmetric, self.is_clipping_in_normalization)

        return log_sepc

    def logspec2ampspec(self, log_spec):
        if self.is_norm:
            log_spec = self._denormalize(log_spec, self.is_symmetric, self.is_clipping_in_normalization)

        log_spec += self.ref_level_db

        amp_spec = np.maximum(self.min_level_amp**2, np.dot(np.power(10.0, log_spec * 0.05), self._inv_mel_basis.T))
        return amp_spec

class VariableNormProcess():
    '''
    Variable, like duration, f0 and bap from world
    '''
    def __init__(self, var_min, var_max, max_abs_value=4.0, is_symmetric=True):
        self.var_min       = var_min
        self.var_max       = var_max
        self.scale         = var_max - var_min
        self.max_abs_value = max_abs_value
        self.is_symmetric  = is_symmetric

        assert self.scale > 0

    def normalize(self, var):
        if self.is_symmetric:
            return np.clip((2 * self.max_abs_value) * ((var - self.var_min) / self.scale) - self.max_abs_value,
                            -self.max_abs_value, self.max_abs_value)
        else:
            return np.clip(self.max_abs_value * ((var - self.var_min) / self.scale), 0, self.max_abs_value)

    def denormalize(self, nvar):
        if self.is_symmetric:
            return (((np.clip(nvar, -self.max_abs_value, self.max_abs_value)
                    + self.max_abs_value) * self.scale / (2 * self.max_abs_value))
                    + self.var_min)
        else:
            return ((np.clip(nvar, 0, self.max_abs_value) * self.scale / self.max_abs_value) + self.var_min)
