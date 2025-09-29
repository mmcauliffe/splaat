import os
import pathlib
import subprocess
from io import BytesIO

import librosa
import pytest
import soundfile


@pytest.fixture(scope="session")
def test_dir():
    base = pathlib.Path(__file__).parent
    return base.joinpath("data")


@pytest.fixture(scope="session")
def audio_dir(test_dir):
    return test_dir.joinpath("audio")


@pytest.fixture(scope="session")
def mfa_mono_path(audio_dir):
    return audio_dir.joinpath("mfa_michael.flac")


@pytest.fixture(scope="session")
def mfa_stereo_path(audio_dir):
    return audio_dir.joinpath("mfa.wav")
