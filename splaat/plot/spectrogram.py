import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram, windows

from splaat.audio.prep import prep_audio


def compute_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    window_size: float = 0.008,
    time_steps: int = 1000,
):
    """Compute a spectrogram from input waveform array of samples.

    Parameters
    ==========
    signal : ndarray
        array of audio samples
    sample_rate : integer
        The sampling frequency of the audio samples in `x`
    window_size : float
        Length in seconds of the analysis window.    For an effective filter bandwidth of 300 Hz use w = 0.008, and for an effective filter bandwidth of 45 Hz use w = 0.04.

    Returns
    =======
    freqs : ndarray
        Array of sample frequencies.
    times : ndarray
        Array of segment times.
    spec : ndarray
        Spectrogram of the audio. By default, the last axis of Sxx corresponds to the segment times.
        It is the magnitude spectrum on the decibel scale, so 20 * log10(Sxx) of the spectrogram
        returned by scipy.signal.spectrogram.


    """
    if signal.ndim > 1:
        signal = signal[0, :]
    x2 = np.rint(32000 * (signal / max(signal))).astype(np.intc)  # scale the signal
    duration = signal.shape[0] / sample_rate
    step = max(duration / time_steps, 0.001)
    order = 13  # FFT size = 2 ^ order

    # set up parameters for signal.spectrogram()
    noverlap = int((window_size - step) * sample_rate)  # skip forward by step between each frame
    nperseg = int(window_size * sample_rate)  # number of samples per waveform window
    nfft = np.power(2, order)  # number of points in the fft
    window = windows.blackmanharris(nperseg)

    freqs, times, spec = spectrogram(
        x2,
        fs=sample_rate,
        noverlap=noverlap,
        window=window,
        nperseg=nperseg,
        nfft=nfft,
        scaling="spectrum",
        mode="magnitude",
        detrend="linear",
    )
    spec = 20 * np.log10(spec + 1)  # put spectrum on decibel scale

    return freqs, times, spec


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    channel: int = 0,
    start: float = 0,
    end: float = -1,
    max_frequency: int = 8000,
    window_size: typing.Union[typing.Literal["wide_band", "narrow_band"], float] = "wide_band",
    preemph: float = 0.94,
    font_size=14,
    min_prop=0.2,
    cmap="Greys",
    figure_height=4.5,
    figure_width=12,
    dpi=72,
):
    """Make pretty good looking spectrograms

    * This function calls scipy.signal.spectrogram to calculate a magnitude spectrogram, which is then transformed to decibels, and passed to plt.imshow for plotting.

    * It mainly is used to produce nice looking figures with features like readable time and frequency axes, scaling so that the time axis is 6.5 inches per second for spectrograms of less than 2 seconds.

    * The function also returns arrays that you can use to create your own figures.

    * The function uses one of two window lengths - 40 msec for narrow band spectrograms, or 8 msec for wideband spectrograms.

    * One option is to add a "spectral slice" to the display - the amplitude/frequency spectrum at a particular point in time.

    Parameters
    ==========
    audio : ndarray
        a one-dimensional array of audio samples.
    sample_rate : int
        The sampling rate of the audio
    channel : int
        The channel of the audio
    start : float, default = 0
        starting time (in seconds) of the waveform chunk to plot -- default plot whole file
    end : float, default = -1
        ending time (in seconds) of the waveform chunk to plot (-1 means go to the end)
    max_frequency : integer, default = 8000
        the top frequency (in Hz) to show in the spectrogram
    window_size : float or {'wide_band', 'narrow_band'}
        effective filter bandwidth of the analysis filter ('wide_band' = 300 Hz, 'narrow_band' = 45 Hz), defaults to wide_band
    preemph : float, default = 0.94
        add high frequency preemphasis before making the spectrogram, a value between 0 and 1
    font_size : float, default = 14
        the font size to use for the axis labels and tick labels.
    min_prop : float, default = 0.2
        set the 'floor' of the gray scale.    The default value specifies that the floor will be
        at 20% of the range between min and max amplitudes.
    figure_height : float, default = 4.5
        Height of the figure
    figure_width : float, default = 12
        Width of the figure
    dpi : int, default = 72
        DPI of the figure
    cmap : string, default = "Grays"
        name of a matplotlib colormap for the spectrogram

    Returns
    =======
    ax : a matplotlib axes object
        The plot axes is returned
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of the audio. By default, the last axis of Sxx corresponds to the segment times.
        It is the magnitude spectrum on the decibel scale, so 20 * log10(Sxx) of the spectrogram
        returned by scipy.signal.spectrogram.

    Examples
    ========

    Plot a spectrogram of a portion of the sound file from 1.5 to 2 seconds.
    Then add a vertical red line at time 1.71

    .. code-block:: Python

        import matplotlib.pyplot as plt

        audio_dir = importlib.resources.files('phonlab') / 'data' / 'example_audio'
        example_file = audio_dir / 'sf3_cln.wav'

        x,fs = phon.loadsig(example_file,chansel=[0])
        phon.sgram(x,fs,start=1.5, end=2.0)
        plt.axvline(1.71,color="red")

    .. figure:: images/burst.png
         :scale: 50 %
         :alt: a spectrogram with a red line marking the location of the burst
         :align: center

         Marking the burst found by `phon.burst()`

    Read a file into an array `x`, track the formant frequencies in the file, use them to produce
    sine wave speech, and then plot a spectrogram of the resulting signal.

    .. code-block:: Python

        example_file = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'sf3_cln.wav'

        x,fs = phon.loadsig(example_file, chansel=[0])
        fmtsdf = phon.track_formants(x,fs)    # track the formants
        x2,fs2 = phon.sine_synth(fmtsdf)     # use the formants to produce sinewave synthesis
        ax1,f,t,Sxx = phon.sgram(x2,fs2, band="nb", preemph=0)    # plot a spectrogram of it

    .. figure:: images/sine_synth.png
         :scale: 40 %
         :alt: a spectrogram of sine-wave synthesis
         :align: center

         Showing the spectrogram of sine-wave synthesis.

    """
    target_sample_rate = (
        max_frequency * 2
    )  # top frequency is the Nyquist frequency for the analysis

    if window_size == "wide_band":
        window_size = 0.008  # analysis window size for wide band spectrogram
    elif window_size == "narrow_band":
        window_size = 0.04  # analysis window size for narrow band spectrogram (sec)

    # set up parameters for the spectrogram window
    cmap = plt.get_cmap(cmap)

    if channel is not None and audio.ndim > 1:
        audio = audio[channel, :]

    # ----------- read and condition waveform -----------------------
    audio, sample_rate = prep_audio(
        audio,
        sample_rate,
        target_sample_rate=target_sample_rate,
        preemph=preemph,
        quiet=True,
    )

    start_sample = int(start * sample_rate)  # index of starting time: seconds to samples
    end_sample = int(end * sample_rate)  # index of ending time
    if end_sample < 0 or end_sample > len(audio):  # stop at the end of the waveform
        end_sample = len(audio)
    if start_sample > end_sample:  # don't let start follow end
        start_sample = 0

    # ----------- compute the spectrogram ---------------------------------
    freqs, times, spec = compute_spectrogram(
        audio[start_sample:end_sample], sample_rate, window_size
    )

    # ------------ display in a matplotlib figure --------------------
    times = np.add(times, start)  # increment the spectrogram time by the start value
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
    ax1 = fig.add_subplot(111)

    vmin = np.min(spec) + (np.max(spec) - np.min(spec)) * min_prop
    extent = (
        min(times),
        max(times),
        min(freqs),
        max(freqs),
    )  # get the time and frequency values for indices.
    _ = ax1.imshow(
        spec,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        extent=extent,
        origin="lower",
    )
    ax1.grid(which="major", axis="y", linestyle=":")  # add grid lines
    ax1.set_xlabel("Time (sec)", size=font_size)
    ax1.set_ylabel("Frequency (Hz)", size=font_size)
    ax1.tick_params(labelsize=font_size)

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    return fig, freqs, times, spec
