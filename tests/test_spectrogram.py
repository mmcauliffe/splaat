from splaat.audio.load import load_audio
from splaat.plot.spectrogram import compute_spectrogram


def test_spectrogram(mfa_mono_path):
    x, fs = load_audio(mfa_mono_path)
    freqs, times, spec = compute_spectrogram(x, fs)
    assert freqs.shape[0] == 4097
    assert times.shape[0] == 983
    assert spec.shape == (4097, 983)
