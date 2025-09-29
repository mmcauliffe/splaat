import numpy as np
import librosa


def load_audio(path, channel=None, begin=0.0, end=None, sample_rate=None, dtype=np.float32):
    duration = None
    if end is not None and end > begin:
        duration = end - begin
    y, fs = librosa.load(
        path, sr=sample_rate, mono=False, offset=begin, duration=duration, dtype=dtype
    )

    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)
    if channel is not None:
        y = y[channel, :]
    return y, fs
