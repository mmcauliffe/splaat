import numpy as np
from librosa.effects import preemphasis
from scipy.signal import resample_poly


def prep_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate=None,
    dither: float = None,
    preemph: float = None,
    scale: float = None,
    output_type: np.dtype = np.float32,
):
    """Prepare an array of audio waveform samples for acoustic analysis.

    Parameters
    ==========
        x : array
            a one-dimensional numpy array with audio samples in it.

        sample_rate : int
              The sampling rate of the sound in **x**.

        target_sample_rate : int, optional
            The desired sampling rate of the audio samples that will be returned by the function.
            No resampling is done by default.

        pre : float, default = 0
            how much high frequency preemphasis to apply (between 0 and 1).

        scale: boolean, default = True
            scale the samples to use the full range for audio samples (based on the peak amplitude in the signal)

        dither: float, default = False
            add a tiny bit of noise to the audio to avoid problematic waveforms with many samples at zero amplitude.

        output_type : np.dtype
            The "int" waveform is 16 bit integers - in the range from [-32768, 32767].
            The "float" waveform is 32 bit floating point numbers - in the range from [-1, 1].


    Returns
    =======
        y : ndarray
            a 1D numpy array with audio samples

        fs : int
            the sampling rate of the audio in **y**.

    """
    if target_sample_rate is None:
        target_sample_rate = sample_rate
    else:
        if target_sample_rate != sample_rate:
            cd = np.gcd(sample_rate, target_sample_rate)  # common denominator
            audio = resample_poly(audio, up=target_sample_rate / cd, down=sample_rate / cd)
    if (np.max(audio) + np.min(audio)) < 0:
        audio = -audio
    if preemph is not None and preemph > 0:
        audio = preemphasis(audio, coef=preemph)
    if scale is not None:
        audio /= np.max(audio) * scale
    if dither is not None:
        audio += (np.random.rand(len(audio)) - 0.5) * dither
    if output_type in {np.int8, np.int16, np.int32, np.int64}:
        audio = np.rint(np.iinfo(output_type).max * audio).astype(output_type)
    return audio, target_sample_rate
