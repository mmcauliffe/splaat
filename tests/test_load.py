from splaat.audio.load import load_audio


def test_load(mfa_mono_path):
    x, fs = load_audio(mfa_mono_path)
    assert x.ndim == 2
    assert x.shape[0] == 1


def test_load_stereo(mfa_stereo_path):
    x, fs = load_audio(mfa_stereo_path)
    assert x.ndim == 2
    assert x.shape[0] == 2
