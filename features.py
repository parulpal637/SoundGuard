"""Audio -> compact feature vector. Pure numpy, so it installs cleanly on Windows on ARM."""
import numpy as np

SR = 16000          # sample rate (Hz)
WIN_SEC = 3         # one analysis window = 3 seconds
N_FFT = 1024
HOP = 512
N_MELS = 32
FMIN, FMAX = 50.0, 8000.0
FEAT_DIM = N_MELS * 2 + 1   # mean + std of each mel band (dB) + CPU load (0..1)

_fb = None


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_fb():
    global _fb
    if _fb is None:
        mels = np.linspace(_hz2mel(FMIN), _hz2mel(FMAX), N_MELS + 2)
        hz = _mel2hz(mels)
        bins = np.fft.rfftfreq(N_FFT, 1.0 / SR)
        fb = np.zeros((N_MELS, len(bins)), np.float32)
        for i in range(N_MELS):
            lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
            up = (bins - lo) / (c - lo)
            down = (hi - bins) / (hi - c)
            fb[i] = np.maximum(0.0, np.minimum(up, down))
        _fb = fb
    return _fb


def log_mel(audio):
    """Return log-mel spectrogram in dB, shape [frames, N_MELS]."""
    a = np.asarray(audio, np.float32)
    a = a - a.mean()
    if len(a) < N_FFT:
        a = np.pad(a, (0, N_FFT - len(a)))
    n = 1 + (len(a) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = a[idx] * np.hanning(N_FFT).astype(np.float32)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = power @ _mel_fb().T
    return 10.0 * np.log10(mel + 1e-12)


def feature_vector(audio, cpu_percent):
    """One 3-second window -> vector of length FEAT_DIM."""
    m = log_mel(audio)
    return np.concatenate([m.mean(0), m.std(0), [cpu_percent / 100.0]]).astype(np.float32)
